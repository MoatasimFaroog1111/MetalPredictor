from __future__ import annotations

import argparse
import json
from pathlib import Path

from metal_predictor.multi_horizon.dataset import MultiHorizonDatasetBuilder
from metal_predictor.multi_horizon.development import Stage3DevelopmentRunner
from metal_predictor.multi_horizon.preregistration import preregistration_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output",
        default="artifacts/multi_horizon_stage3/development_report.json",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    checked_preregistration = json.loads(
        (
            repo_root
            / "research_data/bullionvault_horizons/stage2_preregistration.json"
        ).read_text(encoding="utf-8")
    )
    if checked_preregistration != preregistration_payload():
        raise SystemExit(
            "Stage-2 preregistration changed before Stage-3 development evaluation."
        )

    runner = Stage3DevelopmentRunner(
        dataset_builder=MultiHorizonDatasetBuilder(repo_root=repo_root)
    )
    payload = runner.run_all()
    if payload["historical_test_metrics_read"] is not False:
        raise SystemExit("Stage 3 must not read locked historical-test metrics.")
    if payload["historical_test_predictions_computed"] is not False:
        raise SystemExit("Stage 3 must not predict the locked historical test.")

    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
