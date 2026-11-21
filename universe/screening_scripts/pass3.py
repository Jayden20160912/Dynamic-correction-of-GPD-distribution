import sys, time, requests, pandas as pd
sys.path.insert(0, "/Users/jdzhou/Documents/TQT research project")
import data_fetch
H = {"APCA-API-KEY-ID": data_fetch.get_key("ALPACA_API_KEY_ID"),
     "APCA-API-SECRET-KEY": data_fetch.get_key("ALPACA_API_SECRET_KEY")}
raw = sys.argv[1]
s = pd.concat([pd.read_csv(f"{raw}/sub_{q}.txt", sep="\t", dtype=str) for q in ["2015q4", "2016q1"]])
s = s[s.form.isin(["10-K", "10-Q", "10-K/A", "10-Q/A"])]
pref = s.instance.str.split("-").str[0].str.upper()
pref = sorted({p for p in pref if p.isalpha() and 1 <= len(p) <= 5})
done = set(pd.read_parquet(f"{raw}/jan2016_bars.parquet").symbol.unique()) | set(
    pd.concat([pd.read_csv(f"{raw}/sub_{q}.txt", sep="\t", dtype=str, nrows=0) for q in ["2015q4"]]).columns[:0])
todo = [p for p in pref if p not in done]
print("prefix tickers:", len(pref), "not yet queried with asof=2016:", len(todo))
rows = []
for i in range(0, len(todo), 400):
    params = {"symbols": ",".join(todo[i:i+400]), "timeframe": "1Day", "start": "2016-01-04T00:00:00Z",
              "end": "2016-01-29T23:59:59Z", "limit": 10000, "feed": "sip", "adjustment": "raw", "asof": "2016-01-04"}
    while True:
        for k in range(6):
            r = requests.get("https://data.alpaca.markets/v2/stocks/bars", headers=H, params=params, timeout=120)
            if r.ok: break
            time.sleep(2 * 2 ** k)
        js = r.json()
        for sym, bars in (js.get("bars") or {}).items():
            for b in bars: rows.append((sym, b["t"][:10], b["c"], b["v"]))
        if not js.get("next_page_token"): break
        params["page_token"] = js["next_page_token"]
    time.sleep(0.35)
df = pd.DataFrame(rows, columns=["symbol", "date", "close", "volume"])
df.to_parquet(f"{raw}/jan2016_bars_prefix.parquet")
print("extra symbols with bars:", df.symbol.nunique())
