"""Stable identity for rule-policy behavior and teacher datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from footballworld.policies.rule_based.config import RulePolicyConfig

RULE_POLICY_VERSION = 36
"""Behavioral version of the FootballWorld rule policy.

Increment this value whenever identical public inputs can produce a different
action.  Physics and observation-schema versions are tracked independently.
"""


def policy_config_fingerprint(config: RulePolicyConfig) -> str:
    """Return a canonical SHA-256 identity for a policy configuration."""

    if not isinstance(config, RulePolicyConfig):
        raise TypeError("config must be RulePolicyConfig")
    payload = json.dumps(
        asdict(config),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = ["RULE_POLICY_VERSION", "policy_config_fingerprint"]
