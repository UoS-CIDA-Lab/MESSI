"""JAX-free execution contracts shared by policies and data tooling."""

from __future__ import annotations

RULE_POLICY_EXECUTION_PROFILES = (
    "auto",
    "cpu_causal",
    "gpu_dense",
)
RULE_POLICY_EXECUTION_PROFILES_BY_BACKEND = {
    "cpu": "cpu_causal",
    "gpu": "gpu_dense",
}
RULE_POLICY_EXECUTION_SEMANTICS = "action-and-trace-bit-exact/v1"

__all__ = (
    "RULE_POLICY_EXECUTION_PROFILES",
    "RULE_POLICY_EXECUTION_PROFILES_BY_BACKEND",
    "RULE_POLICY_EXECUTION_SEMANTICS",
)
