from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from footballworld.rendering.renderer import _install_renderer_font_path_cache


def _text_pixels(*, cached: bool) -> np.ndarray:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(3.2, 2.4), dpi=100, facecolor="#0c1622")
    canvas = FigureCanvasAgg(figure)
    figure.text(
        0.5,
        0.5,
        "GK 10:45 HOME 2 : 1 AWAY",
        family="monospace",
        fontsize=10.5,
        weight="bold",
        color="white",
        ha="center",
        va="center",
    )
    canvas.draw()
    if cached:
        assert _install_renderer_font_path_cache(canvas.get_renderer())
    canvas.draw()
    return np.asarray(canvas.buffer_rgba()).copy()


def test_renderer_font_path_cache_preserves_agg_pixels() -> None:
    np.testing.assert_array_equal(_text_pixels(cached=False), _text_pixels(cached=True))


def test_renderer_font_path_cache_resolves_one_property_once(monkeypatch) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.backends import backend_agg
    from matplotlib.font_manager import FontProperties

    renderer = backend_agg.RendererAgg(320, 240, 100)
    original = backend_agg._fontManager._find_fonts_by_props
    calls = 0

    def counted(properties, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(properties, *args, **kwargs)

    monkeypatch.setattr(backend_agg._fontManager, "_find_fonts_by_props", counted)
    assert _install_renderer_font_path_cache(renderer)
    properties = FontProperties(family="sans-serif", weight="bold", size=8.0)
    equivalent_properties = properties.copy()
    first = renderer._prepare_font(properties)
    second = renderer._prepare_font(properties)
    equivalent = renderer._prepare_font(equivalent_properties)

    assert calls == 1
    assert first.fname == second.fname == equivalent.fname


def test_renderer_font_path_cache_fails_closed_without_backend_hook() -> None:
    assert not _install_renderer_font_path_cache(SimpleNamespace())
