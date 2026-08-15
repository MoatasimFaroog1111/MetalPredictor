from __future__ import annotations

from metal_predictor.multi_horizon.selection import (
    DevelopmentCandidateEvidence,
    GateDecision,
    WinnerDecision,
)
from metal_predictor.multi_horizon.stage6_preregistration import stage6_preregistration_payload


class Stage6DevelopmentSelectionGate:
    """Apply only the thresholds committed in the Stage-6 preregistration."""

    baseline_id = "random_walk_zero_return"

    def __init__(self) -> None:
        payload = stage6_preregistration_payload()["development_selection_gate"]
        if not isinstance(payload, dict):
            raise ValueError("Malformed Stage-6 development gate.")
        self._gate = payload

    @property
    def thresholds(self) -> dict[str, object]:
        return dict(self._gate)

    def evaluate(self, evidence: DevelopmentCandidateEvidence) -> GateDecision:
        checks = {
            "beats_random_walk_oof_mae": evidence.candidate_oof_mae < evidence.baseline_oof_mae,
            "minimum_better_folds": evidence.better_folds
            >= int(self._gate["minimum_better_folds"]),
            "bootstrap_ci_lower_positive": evidence.bootstrap_ci_low > 0.0,
            "bootstrap_probability_threshold": evidence.bootstrap_probability_better
            >= float(self._gate["probability_candidate_mae_better_minimum"]),
            "directional_accuracy_floor": evidence.directional_accuracy_delta
            >= float(self._gate["directional_accuracy_delta_minimum"]),
        }
        return GateDecision(
            candidate_id=evidence.candidate_id,
            passed=all(checks.values()),
            checks=checks,
        )

    def choose_winner(
        self,
        evidences: tuple[DevelopmentCandidateEvidence, ...],
        decisions: tuple[GateDecision, ...],
    ) -> WinnerDecision:
        by_id = {evidence.candidate_id: evidence for evidence in evidences}
        passing = [decision.candidate_id for decision in decisions if decision.passed]
        if not passing:
            return WinnerDecision(
                selected_id=self.baseline_id,
                selected_kind="BASELINE",
                reason=str(self._gate["if_no_candidate_passes"]),
            )

        tie_order = list(self._gate["tie_break_order"])
        tie_rank = {candidate_id: index for index, candidate_id in enumerate(tie_order)}
        passing.sort(
            key=lambda candidate_id: (
                -by_id[candidate_id].mae_improvement,
                tie_rank.get(candidate_id, len(tie_rank)),
            )
        )
        return WinnerDecision(
            selected_id=passing[0],
            selected_kind="CANDIDATE",
            reason=str(self._gate["winner_rule"]),
        )
