from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import pyarrow
import sklearn

from metal_predictor.core import PipelineConfig
from metal_predictor.data import ParquetDataLoader
from metal_predictor.future_features import SilverFeatureAssembler
from metal_predictor.historical_stress import (
    HistoricalStressEvaluator,
    LongHistoryStressSuite,
)
from metal_predictor.local_archives import Sha256FileFingerprinter
from metal_predictor.stress_split import AnnualStressConfig, PurgedCalendarYearSplitter
from metal_predictor.targets import NextHourTargetBuilder


DEFAULT_INPUT = Path("data/long_history/XAGUSD_H1_2009_2026_USD_PER_KG.parquet")
DEFAULT_OUTPUT_DIR = Path("data/long_history/stress")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-specification annual expanding stress tests on long XAG history."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    fingerprinter = Sha256FileFingerprinter()
    input_fingerprint = fingerprinter.fingerprint(args.input).as_dict()
    config = PipelineConfig(input_path=args.input)
    hourly = ParquetDataLoader().load(args.input)
    assembler = SilverFeatureAssembler(config)
    featured = assembler.transform(hourly)
    labeled = NextHourTargetBuilder(config.columns).build(featured)

    splitter = PurgedCalendarYearSplitter(
        AnnualStressConfig(
            first_evaluation_year=2012,
            last_evaluation_year=2026,
            min_train_rows=5000,
            skip_insufficient_train_years=True,
        )
    )
    evaluator = HistoricalStressEvaluator(splitter)
    suite = LongHistoryStressSuite(evaluator)
    report, folds, oof = suite.run(labeled, assembler.feature_names)
    report["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit_learn": sklearn.__version__,
    }
    report["input"] = str(args.input)
    report["input_fingerprint"] = input_fingerprint
    report["input_rows"] = len(hourly)
    report["feature_count"] = len(assembler.feature_names)
    report["feature_names"] = list(assembler.feature_names)
    report["future_holdout_files_loaded"] = False
    report["split_policy"] = {
        "first_candidate_evaluation_year": 2012,
        "last_candidate_evaluation_year": 2026,
        "minimum_purged_training_rows": 5000,
        "undertrained_early_years": "skip; never relax the fixed minimum",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "long_history_stress_report.json"
    folds_path = args.output_dir / "annual_stress_folds.csv"
    oof_path = args.output_dir / "annual_stress_oof.parquet"
    manifest_path = args.output_dir / "stress_artifact_manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    folds.to_csv(folds_path, index=False)
    oof.to_parquet(oof_path, index=False)

    manifest = {
        "schema_version": 1,
        "input": input_fingerprint,
        "outputs": [
            fingerprinter.fingerprint(report_path).as_dict(),
            fingerprinter.fingerprint(folds_path).as_dict(),
            fingerprinter.fingerprint(oof_path).as_dict(),
        ],
        "future_holdout_files_loaded": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(_arguments()), indent=2))
