from __future__ import annotations
import numpy as np
import pandas as pd
from metal_predictor.core import ColumnConfig


class StrictLeakageGuard:
    _FORBIDDEN_FEATURE_TOKENS = ("target", "future", "lead", "next_", "shift_minus")

    def __init__(self, columns: ColumnConfig) -> None:
        self._c = columns

    def validate(self, full_frame, splits, feature_names, target_names) -> None:
        self._validate_feature_names(feature_names, target_names)
        self._validate_feature_values(full_frame, feature_names)
        self._validate_target_time(full_frame)
        self._validate_targets(full_frame, target_names)
        self._validate_split_order(splits)

    def _validate_feature_names(self, feature_names, target_names) -> None:
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("Duplicate feature names detected.")
        overlap = set(feature_names) & set(target_names)
        if overlap:
            raise ValueError(f"Targets leaked into features: {sorted(overlap)}")
        for name in feature_names:
            low = name.lower()
            if any(token in low for token in self._FORBIDDEN_FEATURE_TOKENS):
                raise ValueError(f"Suspicious forward-looking feature name: {name}")

    def _validate_feature_values(self, frame, feature_names) -> None:
        missing = [name for name in feature_names if name not in frame.columns]
        if missing:
            raise ValueError(f"Declared features not present: {missing}")
        matrix = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if np.isinf(matrix).any():
            raise ValueError("Feature matrix contains infinite values.")

    def _validate_targets(self, frame, target_names) -> None:
        numeric_names = [name for name in target_names if name != "target_timestamp_utc"]
        numeric = frame.loc[:, numeric_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(numeric).all():
            raise ValueError("Targets contain missing or infinite values.")

    def _validate_target_time(self, frame) -> None:
        feature_ts = pd.to_datetime(frame[self._c.timestamp], utc=True)
        target_ts = pd.to_datetime(frame["target_timestamp_utc"], utc=True)
        if not (target_ts - feature_ts).eq(pd.Timedelta(hours=1)).all():
            raise ValueError("Every target must be exactly one hour after its feature timestamp.")

    def _validate_split_order(self, splits) -> None:
        if set(splits) != {"train", "validation", "test"}:
            raise ValueError("Expected train/validation/test splits.")
        for name, part in splits.items():
            ts = pd.to_datetime(part[self._c.timestamp], utc=True)
            if ts.duplicated().any() or not ts.is_monotonic_increasing:
                raise ValueError(f"{name} timestamps are not unique and chronological.")
        train, val, test = splits["train"], splits["validation"], splits["test"]
        train_target_max = pd.to_datetime(train["target_timestamp_utc"], utc=True).max()
        val_feature_min = pd.to_datetime(val[self._c.timestamp], utc=True).min()
        val_target_max = pd.to_datetime(val["target_timestamp_utc"], utc=True).max()
        test_feature_min = pd.to_datetime(test[self._c.timestamp], utc=True).min()
        if not train_target_max < val_feature_min:
            raise ValueError("Train labels overlap Validation features.")
        if not val_target_max < test_feature_min:
            raise ValueError("Validation labels overlap Test features.")
        sets = [set(pd.to_datetime(part[self._c.timestamp], utc=True)) for part in (train, val, test)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("Timestamp overlap detected between splits.")
