import json, sys, time, requests, pandas as pd
sys.path.insert(0, "/Users/jdzhou/Documents/TQT research project")
import data_fetch
H = {"APCA-API-KEY-ID": data_fetch.get_key("ALPACA_API_KEY_ID"),
     "APCA-API-SECRET-KEY": data_fetch.get_key("ALPACA_API_SECRET_KEY")}
raw = sys.argv[1]
assets = json.load(open(f"{raw}/assets_active.json")) + json.load(open(f"{raw}/assets_inactive.json"))
syms = sorted({a["symbol"] for a in assets if a["symbol"].replace(".", "").isalpha()})
print("candidate symbols:", len(syms))

def get(params):
    for i in range(6):
        r = requests.get("https://data.alpaca.markets/v2/stocks/bars", headers=H, params=params, timeout=120)
        if r.status_code == 200: return r.json()
        time.sleep(2 * 2 ** i)
    raise RuntimeError(r.text[:200])

rows = []
for i in range(0, len(syms), 400):
    params = {"symbols": ",".join(syms[i:i+400]), "timeframe": "1Day", "start": "2016-01-04T00:00:00Z",
              "end": "2016-01-29T23:59:59Z", "limit": 10000, "feed": "sip", "adjustment": "raw",
              "asof": "2016-01-04"}
    while True:
        js = get(params)
        for s, bars in (js.get("bars") or {}).items():
            for b in bars: rows.append((s, b["t"][:10], b["c"], b["v"]))
        if not js.get("next_page_token"): break
        params["page_token"] = js["next_page_token"]
    time.sleep(0.35)
df = pd.DataFrame(rows, columns=["symbol", "date", "close", "volume"])
df.to_parquet(f"{raw}/jan2016_bars.parquet")
print("symbols with Jan-2016 bars:", df.symbol.nunique(), "bars:", len(df))
# asof sanity checks
for s in ["SHLD", "FB", "TWTR"]:
    js = get({"symbols": s, "timeframe": "1Day", "start": "2016-01-04T00:00:00Z", "end": "2024-12-31T00:00:00Z",
              "limit": 10000, "feed": "sip", "adjustment": "all", "asof": "2016-01-04"})
    b = (js.get("bars") or {}).get(s, [])
    print("asof check", s, len(b), b[0]["t"][:10] if b else "-", b[-1]["t"][:10] if b else "-")
