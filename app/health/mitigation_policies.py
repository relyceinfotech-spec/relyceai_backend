"""
Phase 8: Mitigation Policies
Deterministic config adjustments applied when health degrades.

Design rules:
  - Only modifies SAFE parameters (limits, flags)
  - NEVER modifies core planning or strategy logic
  - Applied AFTER circuit opens — not before
  - Reversible when health recovers
"""
from __future__ import annotations

from typing import Optional
from app.health.health_monitor import (
    HEALTHY,
    SANDBOX_UNSTABLE,
    HIGH_NODE_FAILURE,
    REPAIR_SYSTEM_STRESSED,
    ROLLBACK_SPIKE,
)


class MitigationPolicies:
    """
    Applies deterministic config adjustments based on health signal.

    Tracks which mitigations are active so they can be reversed.

    Usage:
        mitigation = MitigationPolicies()
        mitigation.apply(health_signal)
    """

    def __init__(self):
        self.active_mitigations: list = []
        self._original_sandbox_enabled: Optional[bool] = None
        self._original_max_retries: Optional[int] = None

    def apply(self, health_signal: str) -> list:
        """
        Apply mitigations for the given health signal.
        Returns list of actions taken.
        """
        actions = []

        if health_signal == SANDBOX_UNSTABLE:
            actions.extend(self._mitigate_sandbox_unstable())

        if health_signal == REPAIR_SYSTEM_STRESSED:
            actions.extend(self._mitigate_repair_stressed())

        if health_signal == HIGH_NODE_FAILURE:
            actions.extend(self._mitigate_high_failure())

        if health_signal == HEALTHY:
            actions.extend(self._restore_defaults())

        self.active_mitigations = actions
        return actions

    def _mitigate_sandbox_unstable(self) -> list:
        """Disable sandbox to fall back to in-process execution."""
        import app.config as config
        actions = []
        if config.SANDBOX_ENABLED:
            self._original_sandbox_enabled = True
            config.SANDBOX_ENABLED = False
            actions.append("SANDBOX_DISABLED")
        return actions

    def _mitigate_repair_stressed(self) -> list:
        """Reduce repair attempts to lower pressure."""
        from app.agent.tool_executor import MAX_RETRIES
        import app.agent.tool_executor as te
        actions = []
        if te.MAX_RETRIES > 1:
            self._original_max_retries = te.MAX_RETRIES
            te.MAX_RETRIES = 1
            actions.append("MAX_RETRIES_REDUCED")
        return actions

    def _mitigate_high_failure(self) -> list:
        """Log warning but don't modify core parameters."""
        return ["HIGH_FAILURE_LOGGED"]

    def _restore_defaults(self) -> list:
        """Restore original values when health recovers."""
        import app.config as config
        import app.agent.tool_executor as te
        actions = []

        if self._original_sandbox_enabled is not None:
            config.SANDBOX_ENABLED = self._original_sandbox_enabled
            self._original_sandbox_enabled = None
            actions.append("SANDBOX_RESTORED")

        if self._original_max_retries is not None:
            te.MAX_RETRIES = self._original_max_retries
            self._original_max_retries = None
            actions.append("MAX_RETRIES_RESTORED")

        return actions

    def get_active(self) -> list:
        """Return currently active mitigations."""
        return self.active_mitigations


# ============================================
# MODULE SINGLETON
# ============================================

_default_policies: Optional[MitigationPolicies] = None


def get_mitigation_policies() -> MitigationPolicies:
    global _default_policies
    if _default_policies is None:
        _default_policies = MitigationPolicies()
    return _default_policies
