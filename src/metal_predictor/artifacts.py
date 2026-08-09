from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from metal_predictor.core import ColumnConfig

class ParquetArtifactWriter:
    def __init__(self, columns: ColumnConfig) -> None:
        self._c = columns

    def write(self, splits, feature_names, target_names, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for name, frame in splits.items():
            frame.to_parquet(output_dir / f"{name}.parquet", index=False)
            ts = pd.to_datetime(frame[self._c.timestamp], utc=True)
            target_ts = pd.to_datetime(frame["target_timestamp_utc"], utc=True)
            manifest[name] = {
                "rows": int(len(frame)),
                "first_feature_timestamp_utc": ts.min().isoformat(),
                "last_feature_timestamp_utc": ts.max().isoformat(),
                "last_target_timestamp_utc": target_ts.max().isoformat(),
            }
        (output_dir / "feature_manifest.json").write_text(json.dumps({
            "features": list(feature_names),
            "targets": list(target_names),
            "feature_count": len(feature_names),
            "policy": "Causal features only; available at completed bar t."
        }, indent=2), encoding="utf-8")
        (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
