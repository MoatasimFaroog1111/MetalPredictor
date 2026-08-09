from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from metal_predictor.modeling import ModelSpec
from metal_predictor.walk_forward import FoldResult


@dataclass(frozen=True)
class CandidateSummary:
    model_name: str
    family: str
    history_days: int | None
    folds: int
    wf_mae_mean: float
    wf_mae_std: float
    wf_rmse_mean: float
    wf_directional_accuracy_mean: float
    wf_pearson_ic_mean: float | None
    wf_price_mae_mean: float

    def as_dict(self):
        return self.__dict__.copy()


class WalkForwardSelectionPolicy:
    """Selects hyperparameters using Train walk-forward results only."""

    def summarize(self, results: tuple[FoldResult, ...]) -> CandidateSummary:
        if not results:
            raise ValueError("Cannot summarize empty walk-forward results.")
        frame = pd.DataFrame([r.flat_dict() for r in results])
        pearson = frame["pearson_ic"].dropna()
        first = results[0]
        return CandidateSummary(
            first.model_name, first.family, first.history_days, len(results),
            float(frame["mae_return"].mean()), float(frame["mae_return"].std(ddof=0)),
            float(frame["rmse_return"].mean()), float(frame["directional_accuracy"].mean()),
            float(pearson.mean()) if not pearson.empty else None,
            float(frame["price_mae_usd_per_kg"].mean()))

    def family_winners(self, specs: tuple[ModelSpec, ...], summaries: tuple[CandidateSummary, ...]):
        summary_by_name = {s.model_name: s for s in summaries}
        families = {}
        for spec in specs:
            families.setdefault(spec.family, []).append(spec)
        winners = []
        for family in sorted(families):
            winner = min(families[family], key=lambda spec: (
                summary_by_name[spec.name].wf_mae_mean,
                summary_by_name[spec.name].wf_rmse_mean,
                summary_by_name[spec.name].wf_mae_std,
                -summary_by_name[spec.name].wf_directional_accuracy_mean,
                spec.name))
            winners.append(winner)
        return tuple(winners)


@dataclass(frozen=True)
class ValidationResult:
    model_name: str
    family: str
    history_days: int | None
    wf_mae_mean: float
    validation_mae_return: float
    validation_rmse_return: float
    validation_directional_accuracy: float
    validation_pearson_ic: float | None
    validation_price_mae_usd_per_kg: float

    def as_dict(self):
        return self.__dict__.copy()


class FinalModelSelectionPolicy:
    """Chooses among family winners using Validation only; Test is invisible here."""

    def choose(self, rows: tuple[ValidationResult, ...]) -> ValidationResult:
        if not rows:
            raise ValueError("No validation results to select from.")
        return min(rows, key=lambda row: (
            row.validation_mae_return, row.wf_mae_mean, row.validation_rmse_return,
            -row.validation_directional_accuracy, row.model_name))
