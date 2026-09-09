"""Validated external reference contracts for descriptive policy comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

REFERENCE_SCHEMA = "footballworld.policy-reference/1"
COMPARABILITY_CLASSES = {
    "closest_comparable",
    "proxy",
    "diagnostic_neighbor",
}


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate policy reference key: {key}")
        result[key] = value
    return result


def load_policy_reference(path: str | Path) -> dict[str, Any]:
    """Load a small aggregate reference contract and fail closed on ambiguity."""

    source_path = Path(path)
    with source_path.open(encoding="utf-8") as stream:
        reference = json.load(stream, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(reference, dict):
        raise TypeError("policy reference root must be an object")
    if reference.get("schema") != REFERENCE_SCHEMA:
        raise ValueError("policy reference schema is unsupported")
    title = reference.get("title")
    source = reference.get("source")
    metrics = reference.get("metrics")
    cautions = reference.get("cautions", [])
    if not isinstance(title, str) or not title.strip():
        raise TypeError("policy reference title must be a non-empty string")
    if not isinstance(source, dict):
        raise TypeError("policy reference source must be an object")
    if not isinstance(metrics, list) or not metrics:
        raise TypeError("policy reference metrics must be a non-empty list")
    if not isinstance(cautions, list) or not all(
        isinstance(item, str) for item in cautions
    ):
        raise TypeError("policy reference cautions must be a list of strings")

    normalized_metrics = []
    seen: set[str] = set()
    for index, item in enumerate(metrics):
        if not isinstance(item, dict):
            raise TypeError(f"policy reference metric {index} must be an object")
        metric_id = item.get("id")
        label = item.get("label")
        unit = item.get("unit")
        comparability = item.get("comparability")
        if not isinstance(metric_id, str) or not metric_id:
            raise TypeError(f"policy reference metric {index} has no valid id")
        if metric_id in seen:
            raise ValueError(f"duplicate policy reference metric id: {metric_id}")
        seen.add(metric_id)
        if not isinstance(label, str) or not label:
            raise TypeError(f"{metric_id}.label must be a non-empty string")
        if not isinstance(unit, str) or not unit:
            raise TypeError(f"{metric_id}.unit must be a non-empty string")
        if comparability not in COMPARABILITY_CLASSES:
            raise ValueError(f"{metric_id}.comparability is unsupported")
        target = _finite_number(item.get("target"), f"{metric_id}.target")
        baseline = item.get("baseline")
        if baseline is not None:
            baseline = _finite_number(baseline, f"{metric_id}.baseline")
        normalized_metrics.append(
            {
                "id": metric_id,
                "label": label,
                "unit": unit,
                "target": target,
                "baseline": baseline,
                "comparability": comparability,
                "definition": str(item.get("definition", "")),
                "caution": str(item.get("caution", "")),
            }
        )

    return {
        "schema": REFERENCE_SCHEMA,
        "title": title.strip(),
        "source": source,
        "metrics": normalized_metrics,
        "cautions": cautions,
        "reference_file": source_path.name,
    }


def attach_policy_reference(
    report: dict[str, Any],
    reference: dict[str, Any],
) -> None:
    """Attach current values and deltas without manufacturing a global score."""

    current_metrics = report.get("policy_alignment", {}).get("metrics", {})
    comparisons = []
    for target in reference["metrics"]:
        current = current_metrics.get(target["id"], {})
        current_value = current.get("value") if isinstance(current, dict) else None
        current_unit = current.get("unit") if isinstance(current, dict) else None
        if current_value is not None and current_unit != target["unit"]:
            raise ValueError(f"{target['id']} unit disagrees with the policy reference")
        if current_value is not None:
            current_value = _finite_number(current_value, f"{target['id']}.current")
        target_value = float(target["target"])
        comparisons.append(
            {
                **target,
                "current": current_value,
                "delta": (
                    None if current_value is None else current_value - target_value
                ),
                "relative_delta": (
                    None
                    if current_value is None or target_value == 0.0
                    else (current_value - target_value) / abs(target_value)
                ),
                "available": current_value is not None,
            }
        )
    report["external_reference"] = {
        "schema": reference["schema"],
        "title": reference["title"],
        "source": reference["source"],
        "reference_file": reference["reference_file"],
        "comparisons": comparisons,
        "cautions": list(reference["cautions"]),
        "interpretation": (
            "Per-metric descriptive comparison only. No aggregate fidelity score "
            "is produced across unlike estimands."
        ),
    }


__all__ = [
    "COMPARABILITY_CLASSES",
    "REFERENCE_SCHEMA",
    "attach_policy_reference",
    "load_policy_reference",
]
