from __future__ import annotations
from datetime import datetime, timedelta, timezone
import json, math
from typing import Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import pandas as pd
from metal_predictor.precious_metals.contracts import JsonTransport, PreciousMetalInstrument
from metal_predictor.price_normalization import TROY_OZ_PER_KG

class UrllibJsonTransport:
    _BASE_URL = "https://freeserv.dukascopy.com/2.0/"
    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self._timeout=float(timeout_seconds)
        if not math.isfinite(self._timeout) or self._timeout<=0: raise ValueError("timeout_seconds must be finite and positive.")
    def get_json(self, params: Mapping[str, object]) -> object:
        url=f"{self._BASE_URL}?{urlencode({k:str(v) for k,v in params.items()})}"
        try:
            with urlopen(Request(url,headers={"User-Agent":"MetalPredictor-Research/1.0"}),timeout=self._timeout) as response: raw=response.read()
        except OSError: raise RuntimeError("Dukascopy research transport request failed.") from None
        try:return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError,json.JSONDecodeError):raise RuntimeError("Dukascopy research API returned invalid JSON.") from None

class DukascopyHistoricalMetalSource:
    _MAX_API_COUNT=5000; _CHUNK_HOURS=24*180
    def __init__(self,api_key:str,transport:JsonTransport|None=None)->None:
        key=api_key.strip()
        if not key:raise ValueError("Dukascopy API key is required for historical research data.")
        self._api_key=key; self._transport=transport or UrllibJsonTransport(); self._instrument_ids={}
    def fetch_hourly(self,instrument:PreciousMetalInstrument,start_utc:datetime,end_utc:datetime)->pd.DataFrame:
        start=self._exact_hour(start_utc,"start_utc"); end=self._exact_hour(end_utc,"end_utc")
        if end<start:raise ValueError("end_utc must be on or after start_utc.")
        iid=self._resolve_instrument_id(instrument); frames=[]; cursor=start
        while cursor<=end:
            chunk_end=min(end,cursor+timedelta(hours=self._CHUNK_HOURS-1))
            payload=self._transport.get_json({"path":"api/historicalPrices","key":self._api_key,"instrument":iid,"timeFrame":"1hour","count":self._MAX_API_COUNT,"start":int(cursor.timestamp()*1000),"end":int((chunk_end+timedelta(hours=1)-timedelta(milliseconds=1)).timestamp()*1000),"dayStartTime":"UTC","offerSide":"B"})
            frames.append(self._parse_historical_payload(payload,instrument,cursor,chunk_end)); cursor=chunk_end+timedelta(hours=1)
        result=pd.concat(frames,ignore_index=True) if frames else self._empty_frame()
        if result.empty:return result
        result=result.sort_values("timestamp_utc").reset_index(drop=True); self._validate_no_conflicting_duplicates(result)
        return result.drop_duplicates(subset=["timestamp_utc"],keep="first").reset_index(drop=True)
    def _resolve_instrument_id(self,instrument):
        if instrument.dukascopy_name in self._instrument_ids:return self._instrument_ids[instrument.dukascopy_name]
        rows=self._extract_rows(self._transport.get_json({"path":"api/instrumentList","key":self._api_key,"fields":"id,name,pipValue,nameLong"}),("instruments","data","items"))
        for row in rows:
            if isinstance(row,Mapping) and str(row.get("name","")).strip().upper()==instrument.dukascopy_name.upper():
                try:iid=int(row["id"])
                except (KeyError,TypeError,ValueError,OverflowError):raise RuntimeError("Invalid Dukascopy instrument id.") from None
                self._instrument_ids[instrument.dukascopy_name]=iid; return iid
        raise RuntimeError(f"Dukascopy instrument list did not contain {instrument.dukascopy_name}.")
    def _parse_historical_payload(self,payload,instrument,start,end):
        parsed=[]
        for row in self._extract_rows(payload,("data","prices","candles","items")):
            candle=self._parse_candle(row)
            if candle is None:continue
            ts,o,h,l,c=candle
            if not start<=ts<=end:continue
            parsed.append({"timestamp_utc":ts,"open_usd_per_kg":o*TROY_OZ_PER_KG,"high_usd_per_kg":h*TROY_OZ_PER_KG,"low_usd_per_kg":l*TROY_OZ_PER_KG,"close_usd_per_kg":c*TROY_OZ_PER_KG,"open_usd_per_oz":o,"high_usd_per_oz":h,"low_usd_per_oz":l,"close_usd_per_oz":c,"quality_flag":"PROVIDER_H1_BID","source_provider":"Dukascopy","source_symbol":instrument.dukascopy_name,"market_type":"commodity_cfd_cross_feed"})
        return pd.DataFrame(parsed,columns=self._columns()) if parsed else self._empty_frame()
    @classmethod
    def _parse_candle(cls,row):
        if not isinstance(row,Mapping):return None
        try:
            ts=cls._parse_timestamp(cls._first(row,"timestamp","time","date","start","startTime")); vals=tuple(float(cls._first(row,*names)) for names in (("open","openPrice","o"),("high","highPrice","h"),("low","lowPrice","l"),("close","closePrice","c")))
        except (KeyError,TypeError,ValueError,OverflowError):return None
        if ts.minute or ts.second or ts.microsecond:return None
        if not all(math.isfinite(v) and v>0 for v in vals):return None
        o,h,l,c=vals
        if h<l or h<max(o,c) or l>min(o,c):return None
        return ts,o,h,l,c
    @staticmethod
    def _extract_rows(payload,keys):
        if isinstance(payload,list):return list(payload)
        if isinstance(payload,Mapping):
            for key in keys:
                if isinstance(payload.get(key),list):return list(payload[key])
        raise RuntimeError("Dukascopy research API returned an unsupported response shape.")
    @staticmethod
    def _first(row,*keys):
        for key in keys:
            if key in row:return row[key]
        raise KeyError(keys[0])
    @staticmethod
    def _parse_timestamp(value):
        if isinstance(value,(int,float)) and not isinstance(value,bool):
            numeric=float(value); seconds=numeric/1000 if abs(numeric)>=100_000_000_000 else numeric
            if not math.isfinite(numeric):raise ValueError
            return datetime.fromtimestamp(seconds,tz=timezone.utc)
        ts=pd.Timestamp(value)
        if ts.tzinfo is None:raise ValueError
        return ts.tz_convert("UTC").to_pydatetime()
    @staticmethod
    def _validate_no_conflicting_duplicates(frame):
        dup=frame[frame["timestamp_utc"].duplicated(keep=False)]; cols=["open_usd_per_kg","high_usd_per_kg","low_usd_per_kg","close_usd_per_kg"]
        for _,group in dup.groupby("timestamp_utc",sort=False):
            if len(group[cols].drop_duplicates())>1:raise RuntimeError("Dukascopy returned conflicting duplicate hourly candles.")
    @staticmethod
    def _exact_hour(value,name):
        if value.tzinfo is None:raise ValueError(f"{name} must be timezone-aware.")
        utc=value.astimezone(timezone.utc)
        if utc.minute or utc.second or utc.microsecond:raise ValueError(f"{name} must align to an exact UTC hour.")
        return utc
    @staticmethod
    def _columns():return ["timestamp_utc","open_usd_per_kg","high_usd_per_kg","low_usd_per_kg","close_usd_per_kg","open_usd_per_oz","high_usd_per_oz","low_usd_per_oz","close_usd_per_oz","quality_flag","source_provider","source_symbol","market_type"]
    @classmethod
    def _empty_frame(cls):return pd.DataFrame(columns=cls._columns())
