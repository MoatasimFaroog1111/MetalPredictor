from __future__ import annotations

import argparse
import json
from pathlib import Path

from metal_predictor.multi_horizon.dataset import MultiHorizonDatasetBuilder
from metal_predictor.multi_horizon.stage6_development import Stage6DevelopmentRunner
from metal_predictor.multi_horizon.stage6_preregistration import (
    stage6_preregistration_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output",
        default="artifacts/multi_horizon_stage6/development_report.json",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    locked_path = (
        repo_root
        / "research_data/bullionvault_horizons/stage6_preregistration.json"
    )
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    if locked != stage6_preregistration_payload():
        raise SystemExit(
            "Stage-6 preregistration file no longer matches the committed code payload."
        )

    runner = Stage6DevelopmentRunner(
        dataset_builder=MultiHorizonDatasetBuilder(repo_root=repo_root)
    )
    payload = runner.run_all()
    if payload["performance_scope"] != "DEVELOPMENT_ONLY":
        raise SystemExit("Stage 6 development scope changed unexpectedly.")
    if payload["historical_test_metrics_read"] is not False:
        raise SystemExit("Stage 6 must not read locked historical-test metrics.")
    if payload["historical_test_predictions_computed"] is not False:
        raise SystemExit("Stage 6 must not predict the locked historical test.")
    if payload["future_holdout_read"] is not False:
        raise SystemExit("Stage 6 must not read the formal future holdout.")

    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
