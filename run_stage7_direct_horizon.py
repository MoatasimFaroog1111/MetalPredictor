from __future__ import annotations

import argparse
import json
from pathlib import Path

from metal_predictor.direct_horizon.preregistration import stage7_preregistration_payload
from metal_predictor.direct_horizon.research import Stage7DevelopmentRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output",
        default="artifacts/stage7_direct_horizon/development_report.json",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    locked_path = repo_root / "research_data/direct_horizon_stage7/preregistration.json"
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    expected = stage7_preregistration_payload()
    if locked != expected:
        raise SystemExit("Stage-7 preregistration lock no longer matches the code payload.")

    payload = Stage7DevelopmentRunner(repo_root=repo_root).run_all()
    if payload["performance_scope"] != "DEVELOPMENT_ONLY":
        raise SystemExit("Stage-7 performance scope changed unexpectedly.")
    if payload["historical_test_metrics_read"] is not False:
        raise SystemExit("Stage-7 must not compute locked historical-test metrics.")
    if payload["historical_test_predictions_computed"] is not False:
        raise SystemExit("Stage-7 must not predict the locked historical test.")
    if payload["formal_future_holdout_read"] is not False:
        raise SystemExit("Stage-7 must not read the formal future holdout.")
    if payload["candidate_artifacts_written"] is not False:
        raise SystemExit("Stage-7 must not write deployable candidate artifacts.")

    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
