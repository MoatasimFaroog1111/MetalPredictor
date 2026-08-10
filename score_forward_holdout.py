from __future__ import annotations

import argparse
import json
from pathlib import Path

from metal_predictor.future_holdout_scorer import FutureHoldoutScorer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot scorer for the frozen fixed future holdout."
    )
    parser.add_argument("--frozen-root", type=Path, default=Path("."))
    parser.add_argument("--ledger-root", type=Path, default=Path("."))
    args = parser.parse_args()
    report = FutureHoldoutScorer(
        args.frozen_root.resolve(), args.ledger_root.resolve()
    ).score()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
