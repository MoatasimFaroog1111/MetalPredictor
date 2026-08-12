from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class HistoricalBootstrapManifest:
    """Provider-neutral provenance facts for one candidate historical H1 source.

    The manifest describes how a dataset was produced. It deliberately does not inspect
    model targets or future-holdout data and is safe to evaluate before any model fit.
    """

    asset: str
    provider: str
    source_symbol: str
    instrument_semantics: str
    interval: str
    requested_history_days: int
    has_open: bool
    has_high: bool
    has_low: bool
    has_close: bool
    exact_utc_hour_guaranteed: bool
    forward_fill_used: bool
    interpolation_used: bool
    provenance_manifest_embedded: bool
    source_repository: str = ""
    source_commit: str = ""
    source_path: str = ""

    def __post_init__(self) -> None:
        asset = self.asset.strip().upper()
        if asset not in {"XPT", "XPD"}:
            raise ValueError("Historical bootstrap asset must be XPT or XPD.")
        if not self.provider.strip() or not self.source_symbol.strip():
            raise ValueError("Historical bootstrap provider and source symbol are required.")
        if self.interval.strip().lower() not in {"1h", "h1", "hourly"}:
            raise ValueError("Historical bootstrap manifest must describe hourly data.")
        if self.requested_history_days < 1:
            raise ValueError("requested_history_days must be positive.")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HistoricalBootstrapPolicy:
    """Pre-registered source-quality gate for Phase-2 XPT/XPD H1 research."""

    minimum_history_days: int = 365 * 5
    allowed_instrument_semantics: tuple[str, ...] = (
        "SPOT",
        "SPOT_BID",
        "SPOT_ASK",
        "COMMODITY_CFD_CROSS_FEED",
    )
    require_complete_ohlc: bool = True
    require_exact_utc_hour: bool = True
    require_embedded_provenance: bool = True
    forbid_fill_or_interpolation: bool = True

    def __post_init__(self) -> None:
        if self.minimum_history_days < 365:
            raise ValueError("minimum_history_days must be at least one year.")
        if not self.allowed_instrument_semantics:
            raise ValueError("At least one allowed instrument semantic is required.")


@dataclass(frozen=True)
class HistoricalBootstrapAssessment:
    status: str
    asset: str
    provider: str
    source_symbol: str
    failures: tuple[str, ...]
    policy: HistoricalBootstrapPolicy

    @property
    def accepted(self) -> bool:
        return self.status == "PASS"

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "accepted": self.accepted,
            "asset": self.asset,
            "provider": self.provider,
            "source_symbol": self.source_symbol,
            "failures": list(self.failures),
            "policy": asdict(self.policy),
        }


class HistoricalBootstrapProvenanceGate:
    """Fails closed before data import/model fitting when provenance is unsuitable."""

    def __init__(self, policy: HistoricalBootstrapPolicy | None = None) -> None:
        self._policy = policy or HistoricalBootstrapPolicy()

    def assess(self, manifest: HistoricalBootstrapManifest) -> HistoricalBootstrapAssessment:
        failures: list[str] = []
        semantics = manifest.instrument_semantics.strip().upper()

        if semantics not in set(self._policy.allowed_instrument_semantics):
            failures.append("INSTRUMENT_SEMANTICS_NOT_ALLOWED")
        if manifest.requested_history_days < self._policy.minimum_history_days:
            failures.append("INSUFFICIENT_HOURLY_HISTORY")
        if self._policy.require_complete_ohlc and not all(
            (manifest.has_open, manifest.has_high, manifest.has_low, manifest.has_close)
        ):
            failures.append("INCOMPLETE_OHLC_SCHEMA")
        if self._policy.require_exact_utc_hour and not manifest.exact_utc_hour_guaranteed:
            failures.append("EXACT_UTC_HOUR_NOT_GUARANTEED")
        if self._policy.require_embedded_provenance and not manifest.provenance_manifest_embedded:
            failures.append("PROVENANCE_MANIFEST_MISSING")
        if self._policy.forbid_fill_or_interpolation and (
            manifest.forward_fill_used or manifest.interpolation_used
        ):
            failures.append("FILL_OR_INTERPOLATION_FORBIDDEN")

        return HistoricalBootstrapAssessment(
            status="PASS" if not failures else "REJECT",
            asset=manifest.asset.strip().upper(),
            provider=manifest.provider.strip(),
            source_symbol=manifest.source_symbol.strip(),
            failures=tuple(failures),
            policy=self._policy,
        )

    def validate_many(
        self,
        manifests: Iterable[HistoricalBootstrapManifest],
    ) -> tuple[HistoricalBootstrapAssessment, ...]:
        assessments = tuple(self.assess(item) for item in manifests)
        if not assessments:
            raise ValueError("At least one historical bootstrap manifest is required.")
        rejected = [item for item in assessments if not item.accepted]
        if rejected:
            detail = "; ".join(
                f"{item.asset}:{','.join(item.failures)}" for item in rejected
            )
            raise ValueError(f"Historical bootstrap provenance gate rejected source(s): {detail}")
        return assessments
