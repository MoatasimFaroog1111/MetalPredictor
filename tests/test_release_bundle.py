from __future__ import annotations

import pandas as pd

from metal_predictor.release_bundle import PublishedReleaseBundleResolver


def test_same_h15_release_bundle_keeps_latest_actionable_observation() -> None:
    release = pd.Timestamp("2023-08-03 20:15:00", tz="UTC")
    frame = pd.DataFrame({
        "observation_date": pd.to_datetime(["2023-08-01", "2023-08-02", "2023-08-03"]),
        "available_from_utc": [
            release,
            release,
            pd.Timestamp("2023-08-04 20:15:00", tz="UTC"),
        ],
        "rate_2y_percent": [4.90, 4.88, 4.89],
        "rate_10y_percent": [4.05, 4.08, 4.04],
    })

    states, report = PublishedReleaseBundleResolver().resolve(frame)

    assert len(states) == 2
    assert states.loc[0, "observation_date"] == pd.Timestamp("2023-08-02")
    assert states.loc[0, "publication_bundle_size"] == 2
    assert states.loc[0, "rate_2y_percent"] == 4.88
    assert report.input_observations == 3
    assert report.output_public_states == 2
    assert report.bundled_observations == 1
    assert report.multi_observation_releases == 1
    assert report.max_bundle_size == 2


def test_release_bundle_never_changes_single_observation_release() -> None:
    frame = pd.DataFrame({
        "observation_date": pd.to_datetime(["2026-07-27", "2026-07-28"]),
        "available_from_utc": pd.to_datetime([
            "2026-07-28 20:15:00+00:00",
            "2026-07-29 20:15:00+00:00",
        ]),
        "rate_2y_percent": [3.9, 3.91],
        "rate_10y_percent": [4.2, 4.21],
    })
    states, report = PublishedReleaseBundleResolver().resolve(frame)
    assert len(states) == 2
    assert states["publication_bundle_size"].tolist() == [1, 1]
    assert report.bundled_observations == 0
