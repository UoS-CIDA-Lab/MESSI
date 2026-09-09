"""Configuration identity for rule-policy behavior and teacher datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from footballworld.policies.rule_based.config import RulePolicyConfig


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


__all__ = ["policy_config_fingerprint"]
