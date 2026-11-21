import sys, time, requests, numpy as np, pandas as pd
sys.path.insert(0, "/Users/jdzhou/Documents/TQT research project")
import data_fetch
sp = sys.argv[1]
H = {"APCA-API-KEY-ID": data_fetch.get_key("ALPACA_API_KEY_ID"), "APCA-API-SECRET-KEY": data_fetch.get_key("ALPACA_API_SECRET_KEY")}
R = pd.read_parquet(f"{sp}/continuity.parquet"); R = R[R.continues]
out = []
for r in R.itertuples():
    end = pd.Timestamp(r.last_bar)
    p = {"symbols": r.symbol, "timeframe": "1Day", "start": (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d") + "T00:00:00Z",
         "end": "2024-12-31T23:59:59Z", "limit": 10000, "feed": "sip", "adjustment": "all", "asof": "-"}
    bars = []
    while True:
        js = requests.get("https://data.alpaca.markets/v2/stocks/bars", headers=H, params=p, timeout=60).json()
        bars += (js.get("bars") or {}).get(r.symbol, [])
        if not js.get("next_page_token"): break
        p["page_token"] = js["next_page_token"]
    c = np.array([b["c"] for b in bars[:60]]); v = np.array([b["v"] for b in bars[:60]])
    ret = np.diff(np.log(c)) if len(c) > 2 else np.array([0.0])
    out.append((r.symbol, r.name, r.last_bar, len(bars), bars[-1]["t"][:10] if bars else "-",
                round(float(np.std(ret)), 4), int(np.median(v)) if len(v) else 0))
O = pd.DataFrame(out, columns=["symbol", "name", "asof_last_bar", "plain_bars_after", "plain_last_bar", "ret_sd_first60", "median_vol_first60"])
O["genuine_continuation"] = (O.ret_sd_first60 > 0.002) & (O.median_vol_first60 > 10000)
pd.set_option("display.width", 200)
print(O.to_string(index=False))
O.to_parquet(f"{sp}/continuity2.parquet")
