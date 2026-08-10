from __future__ import annotations

import argparse
import json
from pathlib import Path

from metal_predictor.future_holdout_collector import FutureHoldoutCollector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append future XAG observations and frozen predictions without scoring."
    )
    parser.add_argument(
        "--frozen-root",
        type=Path,
        default=Path("."),
        help="Repository snapshot containing frozen code/models/manifest.",
    )
    parser.add_argument(
        "--ledger-root",
        type=Path,
        default=Path("."),
        help="Writable checkout containing forward_holdout ledgers.",
    )
    args = parser.parse_args()
    report = FutureHoldoutCollector(
        args.frozen_root.resolve(), args.ledger_root.resolve()
    ).collect()
    print(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    main()
