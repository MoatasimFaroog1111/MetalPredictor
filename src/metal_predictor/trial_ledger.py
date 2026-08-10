from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrialLedgerEntry:
    stage: str
    counted_trials: int
    rationale: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class ResearchTrialLedger:
    """Explicit conservative count of strategy/configuration trials behind the current candidate.

    DSR is only meaningful when the researcher counts the alternatives tried before
    reporting a winner. This ledger counts unique predeclared configurations/feature
    sets, not repeated folds. Blocked data-source probes that never produced a
    backtest are not counted. Correlation among trials is deliberately NOT used to
    reduce the count, making the multiple-testing correction conservative.
    """

    def __init__(self, initial_model_registry_trials: int) -> None:
        if initial_model_registry_trials < 2:
            raise ValueError("Initial model registry trial count is implausibly small.")
        self._entries = (
            TrialLedgerEntry(
                "initial_model_selection",
                initial_model_registry_trials,
                "All predefined model/baseline configurations in DefaultModelRegistry.",
            ),
            TrialLedgerEntry(
                "cross_asset_feature_sets",
                5,
                "Gold, DXY, Treasury rates, VIX, and CFTC Silver COT feature sets.",
            ),
            TrialLedgerEntry(
                "selective_prediction_coverages",
                5,
                "Predeclared 50%, 35%, 25%, 15%, and 10% target-coverage thresholds.",
            ),
            TrialLedgerEntry(
                "regime_specialist_choices",
                12,
                "Three specialist model candidates across four predeclared regimes.",
            ),
            TrialLedgerEntry(
                "oof_stacking_algorithm",
                1,
                "One fixed three-base-model OOF stack with Ridge(alpha=1) meta learner.",
            ),
            TrialLedgerEntry(
                "publication_timing_methodology_rerun",
                1,
                "Treasury experiment was rerun after correcting bar-start versus decision-time semantics.",
            ),
        )

    @property
    def entries(self) -> tuple[TrialLedgerEntry, ...]:
        return self._entries

    @property
    def total_trials(self) -> int:
        return sum(entry.counted_trials for entry in self._entries)

    def as_dict(self) -> dict[str, object]:
        return {
            "counting_policy": (
                "Raw unique research/configuration trials; no correlation haircut. "
                "Blocked source probes with no backtest are excluded."
            ),
            "entries": [entry.as_dict() for entry in self._entries],
            "total_counted_trials": self.total_trials,
        }
