"""내부용 CPU host 집계 worker. 신뢰한 로컬 객체만 Connection으로 전달한다.

GPU backend가 초기화된 parent를 fork하지 않는다. factory(index, initial)는 import
가능한 module-level callable이며 process(offset, chunk), finish()를 가진 객체를 만든다.
한 process_batch가 끝날 때까지 다음 batch는 허용하지 않는다. 원 closure sink는 이
backend가 실행하지 않으며 호출자가 별도 순차 경로로 유지해야 한다.
"""

import argparse
import importlib
import math
from multiprocessing.connection import Connection
import os
import socket
import subprocess
import sys
import tempfile
import threading
import traceback


class HostWorkerError(RuntimeError):
    """CPU worker의 예외·비정상 종료·응답 timeout."""


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name}은 음이 아닌 Python int여야 한다")
    return value


def _resolve(module, name):
    value = importlib.import_module(module)
    for part in name.split('.'):
        value = getattr(value, part)
    return value


def _factory_reference(factory):
    module = getattr(factory, '__module__', None)
    name = getattr(factory, '__qualname__', None)
    if not callable(factory) or not module or module == '__main__' or not name or '<' in name:
        raise TypeError('factory는 import 가능한 module-level callable이어야 한다')
    if _resolve(module, name) is not factory:
        raise TypeError('factory import identity 불일치')
    return module, name


class HostEvaluationPool:
    """경기 index % workers에 고정 배정하는 bounded 동기 host backend.

    initial_by_match와 chunk는 pickle 가능한 신뢰된 CPU 객체여야 한다. worker 수는
    경기 수 이하로 제한한다. in-flight 입력은 한 chunk batch뿐이며 processor의 누적
    보존량은 caller 책임이다. 결과가 거대하면 그 결과 전송 메모리도 추가로 필요하다.
    """

    def __init__(self, worker_count, factory, initial_by_match, *, timeout_s=300.0):
        if os.name != 'posix' or not hasattr(os, 'sched_getaffinity'):
            raise NotImplementedError('host worker는 POSIX CPU affinity/pass_fds 환경만 지원한다')
        _integer(worker_count, 'worker_count')
        if worker_count == 0 or not isinstance(initial_by_match, dict) or not initial_by_match:
            raise ValueError('양의 worker 수와 비어 있지 않은 경기 초기값 dict가 필요하다')
        for index in initial_by_match:
            _integer(index, 'match index')
        if worker_count > len(initial_by_match):
            raise ValueError('worker 수는 경기 수를 초과할 수 없다')
        cpus = sorted(os.sched_getaffinity(0))
        if worker_count > len(cpus):
            raise ValueError('worker 수는 parent 허용 CPU 수를 초과할 수 없다')
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('timeout_s는 유한 양수여야 한다')
        module, name = _factory_reference(factory)
        self._workers = []
        self._indices = set(initial_by_match)
        self._offsets = {}
        self._lock = threading.Lock()
        self._closed = False
        self._timeout = float(timeout_s)
        self.worker_count = worker_count
        self.worker_cpus = tuple(cpus[:worker_count])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
                   PYTHONDONTWRITEBYTECODE='1')
        try:
            for shard in range(worker_count):
                parent_socket, child_socket = socket.socketpair()
                connection = Connection(parent_socket.detach())
                log = tempfile.TemporaryFile()
                try:
                    child_env = dict(env, MESSI_HOST_WORKER_CPU=str(self.worker_cpus[shard]))
                    # package import보다 먼저 pin한다. preexec_fn/fork 상태 복제를 쓰지 않는다.
                    bootstrap = (
                        "import os,runpy,sys; "
                        "os.sched_setaffinity(0,{int(os.environ['MESSI_HOST_WORKER_CPU'])}); "
                        "sys.argv=['messi.analysis.host_parallel','--worker-fd',sys.argv[1]]; "
                        "runpy.run_module('messi.analysis.host_parallel',run_name='__main__')"
                    )
                    process = subprocess.Popen(
                        [sys.executable, '-c', bootstrap, str(child_socket.fileno())],
                        env=child_env, pass_fds=(child_socket.fileno(),),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    )
                except BaseException:
                    connection.close()
                    log.close()
                    raise
                finally:
                    child_socket.close()
                worker = (process, connection, log)
                self._workers.append(worker)
                connection.send(('initialize', module, name,
                    [(i, payload) for i, payload in initial_by_match.items() if i % worker_count == shard]))
            for worker in self._workers:
                self._receive(worker)
        except BaseException:
            self.close()
            raise

    @property
    def worker_pids(self):
        return tuple(process.pid for process, _, _ in self._workers)

    def _receive(self, worker):
        process, connection, log = worker
        if not connection.poll(self._timeout):
            raise HostWorkerError(f'worker {process.pid} 응답 timeout')
        try:
            status, payload = connection.recv()
        except (EOFError, OSError) as error:
            log.seek(0, 2)
            size = log.tell()
            log.seek(max(0, size - 4096))
            detail = log.read().decode(errors='replace')
            raise HostWorkerError(f'worker {process.pid} 연결 종료: {detail}') from error
        if status != 'ok':
            raise HostWorkerError(f'worker {process.pid} 실패:\n{payload}')
        return payload

    def process_batch(self, entries):
        """[(match_index, offset, host_chunk)] → {index: process 반환값}."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('동시에 두 chunk batch를 제출할 수 없다')
        try:
            if self._closed:
                raise RuntimeError('닫힌 host pool')
            entries = list(entries)
            seen, offsets = set(), dict(self._offsets)
            groups = [[] for _ in self._workers]
            for index, offset, chunk in entries:
                _integer(index, 'match index')
                _integer(offset, 'offset')
                if index not in self._indices or index in seen:
                    raise ValueError('미등록 경기 또는 batch 내 중복 경기')
                if index in offsets and offset <= offsets[index]:
                    raise ValueError('경기 chunk offset은 엄격히 증가해야 한다')
                seen.add(index)
                offsets[index] = offset
                groups[index % self.worker_count].append((index, offset, chunk))
            pending = []
            try:
                for worker, group in zip(self._workers, groups, strict=True):
                    if group:
                        worker[1].send(('process', group))
                        pending.append(worker)
                result = {}
                for worker in pending:
                    result.update(self._receive(worker))
                if set(result) != seen:
                    raise HostWorkerError('worker 결과의 경기 coverage 불일치')
                self._offsets = offsets
                return result
            except BaseException:
                self.close()
                raise
        except BaseException:
            self.close()
            raise
        finally:
            self._lock.release()

    def finish(self):
        """경기별 finish 결과를 모은 뒤 모든 worker를 종료한다."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('chunk 처리 중 finish할 수 없다')
        try:
            if self._closed:
                raise RuntimeError('닫힌 host pool')
            try:
                for _, connection, _ in self._workers:
                    connection.send(('finish',))
                result = {}
                for worker in self._workers:
                    result.update(self._receive(worker))
                if set(result) != self._indices:
                    raise HostWorkerError('worker finish 경기 coverage 불일치')
                return result
            finally:
                self.close()
        finally:
            self._lock.release()

    def close(self):
        """자신이 만든 worker만 종료한다. 반복 호출은 안전하다."""
        if self._closed:
            return
        self._closed = True
        for process, connection, log in self._workers:
            connection.close()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            finally:
                log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _worker(fd):
    if os.environ.get('JAX_PLATFORMS') != 'cpu' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('host worker는 GPU를 노출할 수 없다')
    if os.sched_getaffinity(0) != {int(os.environ['MESSI_HOST_WORKER_CPU'])}:
        raise RuntimeError('host worker의 단일 CPU affinity 불일치')
    connection = Connection(fd)
    try:
        command, module, name, initial = connection.recv()
        if command != 'initialize':
            raise ValueError('worker 초기화 순서 불일치')
        factory = _resolve(module, name)
        processors = {index: factory(index, value) for index, value in initial}
        connection.send(('ok', None))
        while True:
            message = connection.recv()
            if message[0] == 'process':
                result = {index: processors[index].process(offset, chunk)
                          for index, offset, chunk in message[1]}
                connection.send(('ok', result))
            elif message[0] == 'finish':
                connection.send(('ok', {index: p.finish() for index, p in processors.items()}))
                break
            else:
                raise ValueError('알 수 없는 worker command')
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-fd', type=int, required=True)
    _worker(parser.parse_args().worker_fd)
