from __future__ import annotations

import argparse
import json
from pathlib import Path

from metal_predictor.multi_horizon.dataset import MultiHorizonDatasetBuilder
from metal_predictor.multi_horizon.preregistration import (
    preregistration_fingerprint_sha256,
    preregistration_payload,
)
from metal_predictor.multi_horizon.split import ExpandingWalkForwardPlanner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output",
        default="artifacts/multi_horizon_stage2/dataset_report.json",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    builder = MultiHorizonDatasetBuilder(repo_root=repo_root)
    payload = builder.build_report_for_all()
    payload["preregistration"] = {
        "fingerprint_sha256": preregistration_fingerprint_sha256(),
        "performance_metrics_computed": False,
    }

    planner = ExpandingWalkForwardPlanner()
    split_plans: dict[str, object] = {}
    for key in ("4h", "12h", "2d", "30d"):
        dataset, _ = builder.build(key)
        plan = planner.plan(dataset.model_row_count)
        split_plans[key] = {
            "total_rows": plan.total_rows,
            "development_end_exclusive": plan.development_end_exclusive,
            "folds": [
                {
                    "fold_number": fold.fold_number,
                    "train_start": fold.train_start,
                    "train_end_exclusive": fold.train_end_exclusive,
                    "validation_start": fold.validation_start,
                    "validation_end_exclusive": fold.validation_end_exclusive,
                    "purge_bars": fold.purge_bars,
                    "train_row_count": fold.train_row_count,
                    "validation_row_count": fold.validation_row_count,
                }
                for fold in plan.folds
            ],
            "historical_test": {
                "start": plan.historical_test.start,
                "end_exclusive": plan.historical_test.end_exclusive,
                "row_count": plan.historical_test.row_count,
                "metrics_read": False,
            },
        }
    payload["walk_forward_plans"] = split_plans

    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    prereg_path = repo_root / "research_data/bullionvault_horizons/stage2_preregistration.json"
    checked = json.loads(prereg_path.read_text(encoding="utf-8"))
    if checked != preregistration_payload():
        raise SystemExit("Checked Stage-2 preregistration JSON does not match code.")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
