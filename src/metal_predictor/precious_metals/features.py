from __future__ import annotations
import numpy as np
import pandas as pd
from metal_predictor.alignment import ExactTimestampAligner
from metal_predictor.core import ColumnConfig

FEATURE_VERSION="precious-metals-cross-asset-v1"

def _exact_lag(series:pd.Series,timestamps:pd.Series,hours:int)->pd.Series:
    idx=pd.DatetimeIndex(pd.to_datetime(timestamps,utc=True)); keyed=pd.Series(series.to_numpy(dtype=float),index=idx)
    return pd.Series(keyed.reindex(idx-pd.Timedelta(hours=hours)).to_numpy(),index=series.index,dtype=float)

class PlatinumPalladiumCrossAssetFeatures:
    def __init__(self,platinum_frame,palladium_frame,aligner:ExactTimestampAligner,silver_columns:ColumnConfig,return_lags=(1,6,24),ratio_lags=(6,24),corr_windows=(24,72),volatility_window=24):
        if return_lags!=(1,6,24) or ratio_lags!=(6,24) or corr_windows!=(24,72):raise ValueError("Precious-metals v1 feature windows are pre-registered and immutable.")
        if volatility_window!=24:raise ValueError("Precious-metals v1 volatility window is fixed at 24 hours.")
        self._frames={"xpt":self._validate_market(platinum_frame,"XPT"),"xpd":self._validate_market(palladium_frame,"XPD")}; self._aligner=aligner; self._c=silver_columns; self._return_lags=return_lags; self._ratio_lags=ratio_lags; self._corr_windows=corr_windows; self._vol_window=volatility_window; self._names=self._feature_names()
    @property
    def feature_names(self):return self._names
    @property
    def feature_version(self):return FEATURE_VERSION
    def transform(self,frame):
        out=frame.copy(deep=True); ts=pd.to_datetime(out[self._c.timestamp],utc=True,errors="raise"); silver_close=pd.to_numeric(out[self._c.close],errors="coerce").astype(float); silver_log=np.log(silver_close)
        for prefix,market in self._frames.items():
            aux=self._build_auxiliary_table(market,prefix); cols=tuple(c for c in aux.columns if c!="timestamp_utc"); aligned=self._aligner.align(ts,aux,cols)
            for col in aligned.columns:out[col]=aligned[col]
            metal_close=pd.to_numeric(out.pop(f"{prefix}_close_usd_per_kg_internal"),errors="coerce"); out[f"{prefix}_has_exact_current"]=metal_close.notna().astype("int8"); ratio=np.log(metal_close/silver_close); out[f"log_{prefix}_silver_ratio"]=ratio
            for lag in self._return_lags:
                av=f"{prefix}_has_exact_{lag}h"; out[av]=pd.to_numeric(out[av],errors="coerce").fillna(0).astype("int8"); silver_ret=silver_log-_exact_lag(silver_log,ts,lag); metal_ret=pd.to_numeric(out[f"{prefix}_log_return_{lag}h"],errors="coerce"); out[f"{prefix}_silver_relative_return_{lag}h"]=metal_ret-silver_ret
            for lag in self._ratio_lags:out[f"{prefix}_silver_log_ratio_change_{lag}h"]=ratio-_exact_lag(ratio,ts,lag)
            silver_ret=silver_log-_exact_lag(silver_log,ts,1); metal_ret=pd.to_numeric(out[f"{prefix}_log_return_1h"],errors="coerce"); idx=pd.DatetimeIndex(ts); sk=pd.Series(silver_ret.to_numpy(float),index=idx); mk=pd.Series(metal_ret.to_numpy(float),index=idx)
            for window in self._corr_windows:out[f"{prefix}_silver_corr_{window}h"]=sk.rolling(f"{window}h",min_periods=max(4,int(np.ceil(window*.5)))).corr(mk).to_numpy()
        self._add_joint_features(out,ts); return out
    def _build_auxiliary_table(self,market,prefix):
        ts=pd.to_datetime(market["timestamp_utc"],utc=True); close=market["close_usd_per_kg"].astype(float); log_close=np.log(close); o=market["open_usd_per_kg"].astype(float); h=market["high_usd_per_kg"].astype(float); l=market["low_usd_per_kg"].astype(float); safe=o.replace(0,np.nan)
        r=pd.DataFrame({"timestamp_utc":ts,f"{prefix}_close_usd_per_kg_internal":close,f"{prefix}_candle_range_pct":(h-l)/safe,f"{prefix}_candle_body_pct":(close-o)/safe})
        for lag in self._return_lags:
            prior=_exact_lag(log_close,ts,lag); r[f"{prefix}_has_exact_{lag}h"]=prior.notna().astype("int8"); r[f"{prefix}_log_return_{lag}h"]=log_close-prior
        one=log_close-_exact_lag(log_close,ts,1); keyed=pd.Series(one.to_numpy(float),index=pd.DatetimeIndex(ts)); r[f"{prefix}_realized_vol_{self._vol_window}h"]=keyed.rolling(f"{self._vol_window}h",min_periods=max(4,int(np.ceil(self._vol_window*.5)))).std(ddof=0).to_numpy(); return r
    @staticmethod
    def _add_joint_features(out,ts):
        xr=pd.to_numeric(out["log_xpt_silver_ratio"],errors="coerce")-pd.to_numeric(out["log_xpd_silver_ratio"],errors="coerce"); out["both_metals_have_exact_current"]=(out["xpt_has_exact_current"].eq(1)&out["xpd_has_exact_current"].eq(1)).astype("int8"); out["log_xpt_xpd_ratio"]=xr; out["xpt_xpd_log_ratio_change_1h"]=xr-_exact_lag(xr,ts,1)
        a=pd.to_numeric(out["xpt_log_return_1h"],errors="coerce"); b=pd.to_numeric(out["xpd_log_return_1h"],errors="coerce"); both=a.notna()&b.notna(); spread=a-b; out["metal_complex_mean_return_1h"]=((a+b)/2).where(both); out["metal_complex_return_dispersion_1h"]=spread.abs().where(both); out["metal_complex_breadth_1h"]=((np.sign(a)+np.sign(b))/2).where(both); out["xpt_xpd_return_spread_1h"]=spread.where(both)
    def _feature_names(self):
        names=[]
        for p in ("xpt","xpd"):
            names += [f"{p}_has_exact_current",f"{p}_candle_range_pct",f"{p}_candle_body_pct",f"log_{p}_silver_ratio"]
            for lag in self._return_lags:names += [f"{p}_has_exact_{lag}h",f"{p}_log_return_{lag}h",f"{p}_silver_relative_return_{lag}h"]
            for lag in self._ratio_lags:names.append(f"{p}_silver_log_ratio_change_{lag}h")
            names += [f"{p}_realized_vol_{self._vol_window}h"]+[f"{p}_silver_corr_{w}h" for w in self._corr_windows]
        names += ["both_metals_have_exact_current","log_xpt_xpd_ratio","xpt_xpd_log_ratio_change_1h","metal_complex_mean_return_1h","metal_complex_return_dispersion_1h","metal_complex_breadth_1h","xpt_xpd_return_spread_1h"]; return tuple(names)
    @staticmethod
    def _validate_market(frame,asset):
        required={"timestamp_utc","open_usd_per_kg","high_usd_per_kg","low_usd_per_kg","close_usd_per_kg","quality_flag"}; missing=required.difference(frame.columns)
        if missing:raise ValueError(f"{asset} frame missing columns: {sorted(missing)}")
        out=frame.copy(deep=True); out["timestamp_utc"]=pd.to_datetime(out["timestamp_utc"],utc=True,errors="raise"); out=out.sort_values("timestamp_utc").reset_index(drop=True)
        if out["timestamp_utc"].duplicated().any():raise ValueError(f"{asset} timestamps must be unique.")
        if not out["timestamp_utc"].dt.minute.eq(0).all() or not out["timestamp_utc"].dt.second.eq(0).all():raise ValueError(f"{asset} timestamps must align to exact UTC hours.")
        cols=["open_usd_per_kg","high_usd_per_kg","low_usd_per_kg","close_usd_per_kg"]; prices=out[cols].apply(pd.to_numeric,errors="coerce")
        if not np.isfinite(prices.to_numpy(float)).all() or (prices<=0).any().any():raise ValueError(f"{asset} frame contains invalid prices.")
        invalid=prices["high_usd_per_kg"].lt(prices["low_usd_per_kg"])|prices["high_usd_per_kg"].lt(prices[["open_usd_per_kg","close_usd_per_kg"]].max(axis=1))|prices["low_usd_per_kg"].gt(prices[["open_usd_per_kg","close_usd_per_kg"]].min(axis=1))
        if invalid.any():raise ValueError(f"{asset} frame violates OHLC invariants.")
        out[cols]=prices; return out
