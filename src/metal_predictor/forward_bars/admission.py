from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from metal_predictor.forward_bars.contracts import ForwardBar


ADMISSION_POLICY_VERSION: Final = "bullionvault-forward-bar-admission-v1"
MINIMUM_COVERAGE_RATIO: Final = 0.90
REQUIRED_ACCESS_MODE: Final = "AUTHENTICATED_READ_ONLY"
REQUIRED_FRESHNESS_STATUS: Final = "CURRENT_GUI_SOURCE"


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reason: str
    checks: dict[str, bool]

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_version": ADMISSION_POLICY_VERSION,
            "admitted": self.admitted,
            "reason": self.reason,
            "checks": dict(self.checks),
        }


class ForwardBarAdmissionPolicy:
    """Fail-closed gate for forward bars used by research forecast pages.

    Stage 5 admits only high-coverage observations built entirely from authenticated,
    current BullionVault GUI-source snapshots. It never repairs or replaces a rejected
    bar and it never falls back to historical Chart data.
    """

    @property
    def specification(self) -> dict[str, object]:
        return {
            "policy_version": ADMISSION_POLICY_VERSION,
            "minimum_coverage_ratio": MINIMUM_COVERAGE_RATIO,
            "required_access_mode": REQUIRED_ACCESS_MODE,
            "required_freshness_status": REQUIRED_FRESHNESS_STATUS,
            "latest_completed_bar_must_pass": True,
            "historical_chart_fallback_allowed": False,
            "fill_allowed": False,
            "interpolation_allowed": False,
            "synthetic_bar_allowed": False,
        }

    def evaluate(self, bar: ForwardBar) -> AdmissionDecision:
        access_total = sum(int(value) for value in bar.access_mode_counts.values())
        freshness_total = sum(int(value) for value in bar.freshness_status_counts.values())
        authenticated_only = (
            bool(bar.access_mode_counts)
            and set(bar.access_mode_counts) == {REQUIRED_ACCESS_MODE}
            and access_total == bar.snapshot_count
        )
        current_only = (
            bool(bar.freshness_status_counts)
            and set(bar.freshness_status_counts) == {REQUIRED_FRESHNESS_STATUS}
            and freshness_total == bar.snapshot_count
        )
        checks = {
            "coverage_at_least_90_percent": bar.coverage_ratio >= MINIMUM_COVERAGE_RATIO,
            "authenticated_snapshots_only": authenticated_only,
            "current_gui_source_snapshots_only": current_only,
            "minimum_two_observed_snapshots": bar.snapshot_count >= 2,
            "observed_count_not_above_expected": bar.snapshot_count <= bar.expected_snapshot_count,
        }
        admitted = all(checks.values())
        if admitted:
            reason = "ADMITTED_HIGH_COVERAGE_AUTHENTICATED_CURRENT"
        elif not checks["coverage_at_least_90_percent"]:
            reason = "REJECTED_INSUFFICIENT_COVERAGE"
        elif not checks["authenticated_snapshots_only"]:
            reason = "REJECTED_NON_AUTHENTICATED_OR_MIXED_ACCESS"
        elif not checks["current_gui_source_snapshots_only"]:
            reason = "REJECTED_NON_CURRENT_OR_MIXED_FRESHNESS"
        else:
            reason = "REJECTED_OBSERVATION_INTEGRITY"
        return AdmissionDecision(admitted=admitted, reason=reason, checks=checks)
