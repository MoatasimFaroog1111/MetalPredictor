from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ReleaseBundleReport:
    input_observations: int
    output_public_states: int
    bundled_observations: int
    multi_observation_releases: int
    max_bundle_size: int


class PublishedReleaseBundleResolver:
    """Collapses observations first disclosed in the same release into one public state.

    If H.15 publishes multiple previously unavailable observation dates at the same
    timestamp, the state observable immediately after that release is the most recent
    observation in the bundle. Older observations in that same bundle were never
    separately actionable before the later state became public, so treating them as
    sequential feature states would create artificial information timing.
    """

    def resolve(
        self,
        frame: pd.DataFrame,
        observation_name: str = "observation_date",
        available_name: str = "available_from_utc",
    ) -> tuple[pd.DataFrame, ReleaseBundleReport]:
        required = {observation_name, available_name}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Release bundle frame missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("Release bundle frame is empty.")

        out = frame.copy(deep=True)
        out[observation_name] = pd.to_datetime(out[observation_name], errors="raise").dt.normalize()
        out[available_name] = pd.to_datetime(
            out[available_name], utc=True, errors="raise"
        ).astype("datetime64[ns, UTC]")
        out = out.sort_values([available_name, observation_name]).reset_index(drop=True)

        bundle_sizes = out.groupby(available_name, sort=True).size()
        latest = (
            out.groupby(available_name, sort=True, as_index=False)
            .tail(1)
            .copy()
            .sort_values(available_name)
            .reset_index(drop=True)
        )
        latest["publication_bundle_size"] = latest[available_name].map(bundle_sizes).astype("int64")

        if latest[available_name].duplicated().any():
            raise AssertionError("Release bundle resolution failed to create unique publication states.")
        if not latest[available_name].is_monotonic_increasing:
            raise AssertionError("Resolved publication states are not chronological.")

        report = ReleaseBundleReport(
            input_observations=int(len(out)),
            output_public_states=int(len(latest)),
            bundled_observations=int(len(out) - len(latest)),
            multi_observation_releases=int((bundle_sizes > 1).sum()),
            max_bundle_size=int(bundle_sizes.max()),
        )
        return latest, report
