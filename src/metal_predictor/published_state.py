from __future__ import annotations

import pandas as pd


class PublishedStateAligner:
    """Aligns the latest *already published* state to each prediction timestamp.

    Unlike market-price forward filling, this models a piecewise-constant public state:
    a release remains the latest known value until the next publication. The hard
    condition is available_from_utc <= feature timestamp.
    """

    def align(
        self,
        feature_timestamps: pd.Series,
        published: pd.DataFrame,
        value_columns: tuple[str, ...],
        available_name: str = "available_from_utc",
    ) -> pd.DataFrame:
        missing = set((available_name, *value_columns)).difference(published.columns)
        if missing:
            raise ValueError(f"Published state missing columns: {sorted(missing)}")

        feature_ts = pd.to_datetime(feature_timestamps, utc=True, errors="raise").astype(
            "datetime64[ns, UTC]"
        )
        left = pd.DataFrame({
            "__row": feature_timestamps.index,
            "__feature_timestamp": feature_ts,
        }).sort_values("__feature_timestamp")

        right = published.loc[:, [available_name, *value_columns]].copy()
        right[available_name] = pd.to_datetime(
            right[available_name], utc=True, errors="raise"
        ).astype("datetime64[ns, UTC]")
        right = right.sort_values(available_name)
        if right[available_name].duplicated().any():
            raise ValueError("Published-state availability timestamps must be unique.")

        merged = pd.merge_asof(
            left,
            right,
            left_on="__feature_timestamp",
            right_on=available_name,
            direction="backward",
            allow_exact_matches=True,
        )
        observed = merged[available_name].notna()
        if (
            merged.loc[observed, available_name]
            > merged.loc[observed, "__feature_timestamp"]
        ).any():
            raise AssertionError("PublishedStateAligner exposed a future release.")
        merged["published_state_age_hours"] = (
            merged["__feature_timestamp"] - merged[available_name]
        ).dt.total_seconds().div(3600.0)
        merged = merged.sort_values("__row").set_index("__row")
        merged.index.name = feature_timestamps.index.name
        return merged.loc[
            :, [available_name, *value_columns, "published_state_age_hours"]
        ]
