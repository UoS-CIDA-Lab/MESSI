"""Seeded stochastic observation-only baseline policy.

Choices are sampled from explicit, eligibility-masked JAX streams.  A match is
stochastic across seeds and exactly reproducible when its key and inputs match.
"""

from footballworld.policies.rule_based.attack_pattern import AttackPattern
from footballworld.policies.rule_based.config import RulePolicyConfig
from footballworld.policies.rule_based.manager import (
    ManagerPolicyStep,
    RuleBasedManager,
    RuleManagerConfig,
    RuleManagerState,
    initialize_rule_manager_state,
    make_rule_based_manager,
)
from footballworld.policies.rule_based.opening_manager import (
    AuthoredOpeningManagerPolicy,
    AuthoredOpeningSelection,
    RuleBasedOpeningManagerPolicy,
    RuleOpeningManagerConfig,
    RuleOpeningManagerState,
    make_authored_opening_manager_policy,
    make_rule_based_opening_manager_policy,
)
from footballworld.policies.rule_based.policy import (
    PolicyStep,
    RuleBasedPolicy,
    make_rule_based_policy,
)
from footballworld.policies.rule_based.provenance import policy_config_fingerprint
from footballworld.policies.rule_based.state import (
    RulePolicyState,
    apply_tactical_observation,
    initialize_rule_policy_state,
    update_rule_policy_state,
)
from footballworld.policies.rule_based.tactical_plan import (
    TacticalPlan,
    TacticalPlanSelection,
    select_tactical_plans_from_abilities,
    tactical_plan_code,
    tactical_plan_from_code,
)

__all__ = [
    "AttackPattern",
    "AuthoredOpeningManagerPolicy",
    "AuthoredOpeningSelection",
    "ManagerPolicyStep",
    "PolicyStep",
    "RuleBasedManager",
    "RuleBasedOpeningManagerPolicy",
    "RuleBasedPolicy",
    "RuleManagerConfig",
    "RuleManagerState",
    "RuleOpeningManagerConfig",
    "RuleOpeningManagerState",
    "RulePolicyConfig",
    "RulePolicyState",
    "TacticalPlan",
    "TacticalPlanSelection",
    "apply_tactical_observation",
    "initialize_rule_manager_state",
    "initialize_rule_policy_state",
    "make_authored_opening_manager_policy",
    "make_rule_based_manager",
    "make_rule_based_opening_manager_policy",
    "make_rule_based_policy",
    "policy_config_fingerprint",
    "select_tactical_plans_from_abilities",
    "tactical_plan_code",
    "tactical_plan_from_code",
    "update_rule_policy_state",
]
