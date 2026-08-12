from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from metal_predictor.walk_forward import PurgedWalkForwardSplitter


@dataclass(frozen=True)
class PreciousMetalsCoveragePolicy:
    """Pre-registered availability gate applied before any model comparison."""

    min_full_metal_coverage: float = 0.50
    min_train_metal_coverage: float = 0.40
    min_validation_metal_coverage: float = 0.60
    min_validation_joint_coverage: float = 0.50
    min_train_joint_rows: int = 2000
    min_validation_joint_rows: int = 500

    def __post_init__(self) -> None:
        ratios = (
            self.min_full_metal_coverage,
            self.min_train_metal_coverage,
            self.min_validation_metal_coverage,
            self.min_validation_joint_coverage,
        )
        if any(not 0.0 < value <= 1.0 for value in ratios):
            raise ValueError("Coverage ratios must be in (0, 1].")
        if self.min_train_joint_rows < 1 or self.min_validation_joint_rows < 1:
            raise ValueError("Minimum joint-row counts must be positive.")


class PreciousMetalsCoverageValidator:
    """Fails closed when XPT/XPD availability is too sparse for paired evaluation."""

    _REQUIRED = (
        "xpt_has_exact_current",
        "xpd_has_exact_current",
        "both_metals_have_exact_current",
    )

    def __init__(self, policy: PreciousMetalsCoveragePolicy | None = None) -> None:
        self._policy = policy or PreciousMetalsCoveragePolicy()

    def validate(
        self,
        development: pd.DataFrame,
        splitter: PurgedWalkForwardSplitter,
    ) -> dict[str, object]:
        missing = set(self._REQUIRED).difference(development.columns)
        if missing:
            raise ValueError(f"Precious-metals coverage gate missing columns: {sorted(missing)}")
        if development.empty:
            raise ValueError("Precious-metals coverage gate received empty development data.")

        full = self._coverage(development)
        if full["xpt"] < self._policy.min_full_metal_coverage:
            raise ValueError("XPT full-development exact coverage is below the pre-registered minimum.")
        if full["xpd"] < self._policy.min_full_metal_coverage:
            raise ValueError("XPD full-development exact coverage is below the pre-registered minimum.")

        fold_reports: list[dict[str, object]] = []
        for fold in splitter.split(development):
            train = self._coverage(fold.train)
            validation = self._coverage(fold.validation)
            train_joint_rows = int(fold.train["both_metals_have_exact_current"].eq(1).sum())
            validation_joint_rows = int(
                fold.validation["both_metals_have_exact_current"].eq(1).sum()
            )
            failures: list[str] = []
            for metal in ("xpt", "xpd"):
                if train[metal] < self._policy.min_train_metal_coverage:
                    failures.append(f"train_{metal}_coverage")
                if validation[metal] < self._policy.min_validation_metal_coverage:
                    failures.append(f"validation_{metal}_coverage")
            if validation["joint"] < self._policy.min_validation_joint_coverage:
                failures.append("validation_joint_coverage")
            if train_joint_rows < self._policy.min_train_joint_rows:
                failures.append("train_joint_rows")
            if validation_joint_rows < self._policy.min_validation_joint_rows:
                failures.append("validation_joint_rows")
            fold_reports.append({
                "fold": fold.number,
                "train_rows": len(fold.train),
                "validation_rows": len(fold.validation),
                "train_coverage": train,
                "validation_coverage": validation,
                "train_joint_rows": train_joint_rows,
                "validation_joint_rows": validation_joint_rows,
                "passed": not failures,
                "failures": failures,
            })
            if failures:
                raise ValueError(
                    f"Precious-metals coverage gate failed fold {fold.number}: {failures}"
                )

        return {
            "status": "PASS",
            "policy_fixed_before_result": True,
            "policy": asdict(self._policy),
            "full_development_coverage": full,
            "folds": fold_reports,
        }

    @staticmethod
    def _coverage(frame: pd.DataFrame) -> dict[str, float]:
        size = len(frame)
        return {
            "xpt": float(frame["xpt_has_exact_current"].eq(1).sum() / size),
            "xpd": float(frame["xpd_has_exact_current"].eq(1).sum() / size),
            "joint": float(frame["both_metals_have_exact_current"].eq(1).sum() / size),
        }
