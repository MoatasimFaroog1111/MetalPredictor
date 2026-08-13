from datetime import datetime, timedelta, timezone
from metal_predictor.forward_bars.contracts import QuoteSample
from metal_predictor.forward_bars.factory import ForwardBarFactory
from metal_predictor.forward_bars.repository import SQLiteForwardBarRepository

class S:
    def __init__(self, xs): self.xs=xs
    def first_sample_at(self, **_): return self.xs[0].captured_at_utc
    def samples_between(self, start_utc, end_utc, **_): return [x for x in self.xs if start_utc <= x.captured_at_utc < end_utc]

def test_direct_daily_forward_bar(tmp_path):
    u=timezone.utc; d=datetime(2026,8,13,tzinfo=u)
    xs=[QuoteSample("BullionVault","AGXLN","USD",d+timedelta(minutes=1),"AUTHENTICATED_READ_ONLY","CURRENT",2000,2002),QuoteSample("BullionVault","AGXLN","USD",d+timedelta(hours=23,minutes=59),"AUTHENTICATED_READ_ONLY","CURRENT",2010,2012)]
    r=SQLiteForwardBarRepository(tmp_path/"b.sqlite3")
    f=ForwardBarFactory(S(xs),r,security_id="AGXLN",currency="USD",source_cadence_seconds=60,close_delay_seconds=120)
    f.materialize_horizon("1d",now_utc=d+timedelta(days=1,minutes=3))
    b=r.latest_bar("1d"); assert b is not None and b.interval_seconds==86400
