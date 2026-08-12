from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping


ABLATION_VERSION = "precious-metals-family-ablation-v1"


@dataclass(frozen=True)
class PreciousMetalsFeatureFamily:
    """One immutable semantic family within the 43 XPT/XPD candidate features."""

    name: str
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Feature-family name must be non-empty.")
        if not self.features:
            raise ValueError(f"Feature family {self.name} must contain at least one feature.")
        if len(set(self.features)) != len(self.features):
            raise ValueError(f"Feature family {self.name} contains duplicate features.")
        if any(not feature.strip() for feature in self.features):
            raise ValueError(f"Feature family {self.name} contains an empty feature name.")


DEFAULT_FAMILIES: tuple[PreciousMetalsFeatureFamily, ...] = (
    PreciousMetalsFeatureFamily(
        "availability",
        (
            "xpt_has_exact_current",
            "xpt_has_exact_1h",
            "xpt_has_exact_6h",
            "xpt_has_exact_24h",
            "xpd_has_exact_current",
            "xpd_has_exact_1h",
            "xpd_has_exact_6h",
            "xpd_has_exact_24h",
            "both_metals_have_exact_current",
        ),
    ),
    PreciousMetalsFeatureFamily(
        "candle_shape",
        (
            "xpt_candle_range_pct",
            "xpt_candle_body_pct",
            "xpd_candle_range_pct",
            "xpd_candle_body_pct",
        ),
    ),
    PreciousMetalsFeatureFamily(
        "own_returns",
        (
            "xpt_log_return_1h",
            "xpt_log_return_6h",
            "xpt_log_return_24h",
            "xpd_log_return_1h",
            "xpd_log_return_6h",
            "xpd_log_return_24h",
        ),
    ),
    PreciousMetalsFeatureFamily(
        "silver_relative_momentum",
        (
            "xpt_silver_relative_return_1h",
            "xpt_silver_relative_return_6h",
            "xpt_silver_relative_return_24h",
            "xpd_silver_relative_return_1h",
            "xpd_silver_relative_return_6h",
            "xpd_silver_relative_return_24h",
        ),
    ),
    PreciousMetalsFeatureFamily(
        "silver_ratios",
        (
            "log_xpt_silver_ratio",
            "xpt_silver_log_ratio_change_6h",
            "xpt_silver_log_ratio_change_24h",
            "log_xpd_silver_ratio",
            "xpd_silver_log_ratio_change_6h",
            "xpd_silver_log_ratio_change_24h",
        ),
    ),
    PreciousMetalsFeatureFamily(
        "volatility_correlation",
        (
            "xpt_realized_vol_24h",
            "xpt_silver_corr_24h",
            "xpt_silver_corr_72h",
            "xpd_realized_vol_24h",
            "xpd_silver_corr_24h",
            "xpd_silver_corr_72h",
        ),
    ),
    PreciousMetalsFeatureFamily(
        "cross_metal_complex",
        (
            "log_xpt_xpd_ratio",
            "xpt_xpd_log_ratio_change_1h",
            "metal_complex_mean_return_1h",
            "metal_complex_return_dispersion_1h",
            "metal_complex_breadth_1h",
            "xpt_xpd_return_spread_1h",
        ),
    ),
)


class PreciousMetalsFeatureFamilyRegistry:
    """Fixed semantic partition used before reading any family-ablation result."""

    def __init__(
        self,
        families: tuple[PreciousMetalsFeatureFamily, ...] = DEFAULT_FAMILIES,
    ) -> None:
        if not families:
            raise ValueError("At least one feature family is required.")
        names = [family.name for family in families]
        if len(set(names)) != len(names):
            raise ValueError("Feature-family names must be unique.")
        all_features = [feature for family in families for feature in family.features]
        if len(set(all_features)) != len(all_features):
            raise ValueError("Feature families must form a non-overlapping partition.")
        self._families = families
        self._all_features = tuple(all_features)

    @property
    def version(self) -> str:
        return ABLATION_VERSION

    @property
    def families(self) -> tuple[PreciousMetalsFeatureFamily, ...]:
        return self._families

    @property
    def all_features(self) -> tuple[str, ...]:
        return self._all_features

    def validate_candidate_features(self, candidate_features: tuple[str, ...]) -> None:
        if len(candidate_features) != len(set(candidate_features)):
            raise ValueError("Candidate feature list contains duplicates.")
        expected = set(self._all_features)
        observed = set(candidate_features)
        missing = sorted(expected.difference(observed))
        unexpected = sorted(observed.difference(expected))
        if missing or unexpected:
            raise ValueError(
                "Ablation registry does not exactly partition candidate features; "
                f"missing={missing}, unexpected={unexpected}."
            )

    def fingerprint(self) -> str:
        payload = {
            "version": self.version,
            "families": [
                {"name": family.name, "features": list(family.features)}
                for family in self._families
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FamilyEvidencePolicy:
    """Conservative, pre-registered family-retention rule for exploratory ablation.

    Bootstrap probabilities are not treated as classical p-values. The strong gate uses
    a conservative familywise probability threshold of 1 - alpha / family_count only as
    an additional guard against repeatedly inspecting seven semantic families.
    """

    family_count: int = 7
    familywise_alpha: float = 0.05
    supportive_probability: float = 0.90
    supportive_min_better_folds: int = 3
    strong_min_better_folds: int = 4

    def __post_init__(self) -> None:
        if self.family_count < 1:
            raise ValueError("family_count must be positive.")
        if not 0 < self.familywise_alpha < 1:
            raise ValueError("familywise_alpha must be between zero and one.")
        if not 0.5 < self.supportive_probability < 1:
            raise ValueError("supportive_probability must be between 0.5 and one.")
        if not 1 <= self.supportive_min_better_folds <= self.strong_min_better_folds:
            raise ValueError("Fold thresholds are inconsistent.")

    @property
    def strong_probability(self) -> float:
        return 1.0 - self.familywise_alpha / self.family_count

    def assess(
        self,
        family: PreciousMetalsFeatureFamily,
        addon_report: Mapping[str, object],
        conditional_report: Mapping[str, object],
    ) -> dict[str, object]:
        addon = self._extract(addon_report)
        conditional = self._extract(conditional_report)

        addon_strong = self._strong(addon)
        conditional_strong = self._strong(conditional)
        addon_supportive = self._supportive(addon)
        conditional_supportive = self._supportive(conditional)
        conditional_harmful = self._strong_harm(conditional)

        if conditional_harmful:
            classification = "HARMFUL_CONDITIONAL"
            retain = False
        elif addon_strong and conditional_strong:
            classification = "CORE"
            retain = True
        elif conditional_strong:
            classification = "COMPLEMENTARY_CORE"
            retain = True
        elif addon_strong and conditional_supportive:
            classification = "STRONG_STANDALONE_SUPPORTIVE_CONDITIONAL"
            retain = True
        elif addon_supportive and conditional_supportive:
            classification = "SUPPORTIVE"
            retain = True
        elif addon_strong:
            classification = "STANDALONE_BUT_REDUNDANT_IN_FULL_SET"
            retain = False
        else:
            classification = "REDUNDANT_OR_UNRESOLVED"
            retain = False

        return {
            "family": family.name,
            "feature_count": len(family.features),
            "features": list(family.features),
            "classification": classification,
            "retain_in_parsimonious_candidate": retain,
            "addon_evidence": addon,
            "conditional_evidence": conditional,
            "thresholds": {
                "strong_probability": self.strong_probability,
                "supportive_probability": self.supportive_probability,
                "strong_min_better_folds": self.strong_min_better_folds,
                "supportive_min_better_folds": self.supportive_min_better_folds,
                "strong_requires_positive_ci95_low": True,
                "strong_harm_requires_negative_ci95_high": True,
            },
        }

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["strong_probability"] = self.strong_probability
        payload["probability_gate_note"] = (
            "Conservative familywise probability gate; bootstrap probabilities are not classical p-values."
        )
        return payload

    def _strong(self, stats: Mapping[str, object]) -> bool:
        return (
            float(stats["mae_improvement"]) > 0
            and int(stats["better_folds"]) >= self.strong_min_better_folds
            and float(stats["ci95_low"]) > 0
            and float(stats["probability_better"]) >= self.strong_probability
        )

    def _supportive(self, stats: Mapping[str, object]) -> bool:
        return (
            float(stats["mae_improvement"]) > 0
            and int(stats["better_folds"]) >= self.supportive_min_better_folds
            and float(stats["probability_better"]) >= self.supportive_probability
        )

    def _strong_harm(self, stats: Mapping[str, object]) -> bool:
        return (
            float(stats["mae_improvement"]) < 0
            and int(stats["better_folds"]) <= 1
            and float(stats["ci95_high"]) < 0
            and float(stats["probability_better"]) <= 1.0 - self.strong_probability
        )

    @staticmethod
    def _extract(report: Mapping[str, object]) -> dict[str, object]:
        bootstrap = report["paired_block_bootstrap"]
        walk_forward = report["walk_forward"]
        if not isinstance(bootstrap, Mapping) or not isinstance(walk_forward, Mapping):
            raise ValueError("Ablation comparison report has an invalid structure.")
        return {
            "mae_improvement": float(bootstrap["mae_improvement_vs_baseline"]),
            "mae_improvement_percent": float(bootstrap["mae_improvement_percent"]),
            "ci95_low": float(bootstrap["improvement_ci95_low"]),
            "ci95_high": float(bootstrap["improvement_ci95_high"]),
            "probability_better": float(bootstrap["probability_selected_better"]),
            "better_folds": int(walk_forward["enhanced_better_folds"]),
            "total_folds": int(walk_forward["folds"]),
        }
