import json, sys, requests
sys.path.insert(0, "/Users/jdzhou/Documents/TQT research project")
import data_fetch
H = {"APCA-API-KEY-ID": data_fetch.get_key("ALPACA_API_KEY_ID"),
     "APCA-API-SECRET-KEY": data_fetch.get_key("ALPACA_API_SECRET_KEY")}
raw = sys.argv[1]
for status in ("active", "inactive"):
    r = requests.get("https://paper-api.alpaca.markets/v2/assets",
                     headers=H, params={"status": status, "asset_class": "us_equity"}, timeout=120)
    print(status, r.status_code, len(r.json()) if r.ok else r.text[:200])
    if r.ok:
        json.dump(r.json(), open(f"{raw}/assets_{status}.json", "w"))
# do delisted symbols have historical bars?
for sym in ["TWTR", "ATVI", "XLNX", "CELG", "SIVB", "FRC", "BBBY", "SHLD"]:
    r = requests.get("https://data.alpaca.markets/v2/stocks/bars", headers=H,
                     params={"symbols": sym, "timeframe": "1Day", "start": "2016-01-01T00:00:00Z",
                             "end": "2024-12-31T00:00:00Z", "limit": 10000, "feed": "sip", "adjustment": "all"}, timeout=60)
    bars = (r.json().get("bars") or {}).get(sym, []) if r.ok else []
    print(sym, r.status_code, len(bars), bars[0]["t"][:10] if bars else "-", bars[-1]["t"][:10] if bars else "-")
