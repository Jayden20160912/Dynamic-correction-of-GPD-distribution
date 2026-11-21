import sys, time, requests, pandas as pd
sys.path.insert(0, "/Users/jdzhou/Documents/TQT research project")
import data_fetch
sp = sys.argv[1]
H = {"APCA-API-KEY-ID": data_fetch.get_key("ALPACA_API_KEY_ID"), "APCA-API-SECRET-KEY": data_fetch.get_key("ALPACA_API_SECRET_KEY")}
S = pd.read_parquet(f"{sp}/selected.parquet")
def bars(sym, start, end, asof):
    p = {"symbols": sym, "timeframe": "1Day", "start": start + "T00:00:00Z", "end": end + "T23:59:59Z",
         "limit": 10000, "feed": "sip", "adjustment": "all", "asof": asof}
    out = []
    while True:
        for k in range(6):
            r = requests.get("https://data.alpaca.markets/v2/stocks/bars", headers=H, params=p, timeout=60)
            if r.ok: break
            time.sleep(2 * 2 ** k)
        js = r.json(); out += (js.get("bars") or {}).get(sym, [])
        if not js.get("next_page_token"): return out
        p["page_token"] = js["next_page_token"]
res = []
for row in S[S.delisted_before_2025].itertuples():
    end = pd.Timestamp(row.last_bar)
    a = bars(row.symbol_2016, (end - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), (end + pd.Timedelta(days=21)).strftime("%Y-%m-%d"), "-")
    before = [b for b in a if b["t"][:10] <= row.last_bar]; after = [b for b in a if b["t"][:10] > row.last_bar]
    jump = (after[0]["c"] / before[-1]["c"] - 1) if before and after else None
    gap = (pd.Timestamp(after[0]["t"][:10]) - end).days if after else None
    res.append((row.symbol_2016, row.name[:32], row.last_bar, len(after), gap, None if jump is None else round(jump, 3)))
R = pd.DataFrame(res, columns=["symbol", "name", "last_bar", "bars_after_plain_ticker", "gap_days", "next_close_jump"])
R["continues"] = (R.bars_after_plain_ticker > 5) & (R.gap_days <= 5) & (R.next_close_jump.abs() < 0.25)
pd.set_option("display.width", 200)
print(R[R.bars_after_plain_ticker > 0].to_string(index=False))
print("\ncontinues under the plain ticker (mapping break, not a delisting):", R.continues.sum(), "of", len(R))
R.to_parquet(f"{sp}/continuity.parquet")
