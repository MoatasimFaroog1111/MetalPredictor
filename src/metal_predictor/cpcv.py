from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CPCVConfig:
    n_groups: int = 6
    test_groups: int = 2
    embargo_rows: int = 1

    def __post_init__(self) -> None:
        if self.n_groups < 4:
            raise ValueError("CPCV requires at least four chronological groups.")
        if not 1 <= self.test_groups < self.n_groups:
            raise ValueError("test_groups must be between 1 and n_groups-1.")
        if self.embargo_rows < 0:
            raise ValueError("embargo_rows cannot be negative.")


@dataclass(frozen=True)
class CPCVSplit:
    number: int
    train: pd.DataFrame
    test: pd.DataFrame
    test_group_ids: tuple[int, ...]
    purged_train_rows: int
    embargoed_train_rows: int


class CombinatorialPurgedSplitter:
    """Combinatorial chronological groups with label-overlap purge and post-test embargo.

    This is a robustness diagnostic, not a live-deployment estimator: training is the
    complement of selected test groups and can therefore contain observations that
    occur after an earlier test group. Purging guarantees that no training label
    interval overlaps a held-out test interval. Live chronology remains governed by
    the separate PurgedWalkForwardSplitter.
    """

    def __init__(
        self,
        config: CPCVConfig,
        timestamp_name: str = "timestamp_utc",
        target_timestamp_name: str = "target_timestamp_utc",
    ) -> None:
        self._config = config
        self._timestamp = timestamp_name
        self._target_timestamp = target_timestamp_name

    @property
    def expected_splits(self) -> int:
        from math import comb
        return comb(self._config.n_groups, self._config.test_groups)

    def split(self, frame: pd.DataFrame) -> tuple[CPCVSplit, ...]:
        required = {self._timestamp, self._target_timestamp}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"CPCV frame missing columns: {sorted(missing)}")
        ordered = frame.sort_values(self._timestamp).reset_index(drop=True).copy()
        ordered[self._timestamp] = pd.to_datetime(
            ordered[self._timestamp], utc=True, errors="raise"
        )
        ordered[self._target_timestamp] = pd.to_datetime(
            ordered[self._target_timestamp], utc=True, errors="raise"
        )
        if ordered.empty or ordered[self._timestamp].duplicated().any():
            raise ValueError("CPCV requires non-empty unique chronological timestamps.")
        if not ordered[self._timestamp].is_monotonic_increasing:
            raise ValueError("CPCV timestamps are not chronological.")

        positions = np.arange(len(ordered), dtype=int)
        groups = tuple(np.asarray(group, dtype=int) for group in np.array_split(positions, self._config.n_groups))
        if any(len(group) < 100 for group in groups):
            raise ValueError("CPCV groups are too small for robust evaluation.")

        results: list[CPCVSplit] = []
        for number, selected_groups in enumerate(
            combinations(range(self._config.n_groups), self._config.test_groups), start=1
        ):
            test_positions = np.concatenate([groups[index] for index in selected_groups])
            test_mask = np.zeros(len(ordered), dtype=bool)
            test_mask[test_positions] = True
            train_mask = ~test_mask
            before_purge = int(train_mask.sum())

            for group_id in selected_groups:
                group_positions = groups[group_id]
                test_start = ordered.loc[group_positions[0], self._timestamp]
                test_end = ordered.loc[group_positions[-1], self._target_timestamp]
                overlap = (
                    ordered[self._timestamp].le(test_end)
                    & ordered[self._target_timestamp].ge(test_start)
                    & train_mask
                )
                train_mask[overlap.to_numpy()] = False

            after_purge = int(train_mask.sum())
            purged = before_purge - after_purge
            embargoed = 0
            if self._config.embargo_rows:
                for group_id in selected_groups:
                    last_position = int(groups[group_id][-1])
                    start = last_position + 1
                    end = min(len(ordered), start + self._config.embargo_rows)
                    for position in range(start, end):
                        if train_mask[position]:
                            train_mask[position] = False
                            embargoed += 1

            train = ordered.loc[train_mask].copy().reset_index(drop=True)
            test = ordered.loc[test_mask].copy().sort_values(self._timestamp).reset_index(drop=True)
            if train.empty or test.empty:
                raise ValueError("CPCV produced an empty train or test partition.")
            train_ts = set(train[self._timestamp])
            test_ts = set(test[self._timestamp])
            if train_ts & test_ts:
                raise AssertionError("CPCV train/test timestamp overlap detected.")
            results.append(CPCVSplit(
                number=number,
                train=train,
                test=test,
                test_group_ids=tuple(selected_groups),
                purged_train_rows=purged,
                embargoed_train_rows=embargoed,
            ))
        if len(results) != self.expected_splits:
            raise AssertionError("Unexpected CPCV split count.")
        return tuple(results)
