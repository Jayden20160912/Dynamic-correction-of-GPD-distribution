import sys, time, requests, numpy as np, pandas as pd
sys.path.insert(0, "/Users/jdzhou/Documents/TQT research project")
import data_fetch
sp = sys.argv[1]
P = pd.read_parquet(f"{sp}/eligible.parquet")

def sector(sic):
    s = int(sic)
    if 1300 <= s <= 1399 or 2900 <= s <= 2999: return "Energy"
    if 4900 <= s <= 4999: return "Utilities"
    if s == 6798 or 6500 <= s <= 6599: return "Real Estate"
    if 6000 <= s <= 6799: return "Financials"
    if 2830 <= s <= 2836 or 3840 <= s <= 3851 or 8000 <= s <= 8099 or s == 5122: return "Health Care"
    if 3570 <= s <= 3579 or 3600 <= s <= 3699 or 3820 <= s <= 3829 or 7370 <= s <= 7379: return "Information Technology"
    if 4800 <= s <= 4899 or 2710 <= s <= 2741 or 7810 <= s <= 7841: return "Communication Services"
    if 2000 <= s <= 2199 or 5400 <= s <= 5499 or 2840 <= s <= 2844 or 5140 <= s <= 5149 or 5912 == s: return "Consumer Staples"
    if 1000 <= s <= 1499 or 2600 <= s <= 2699 or 2800 <= s <= 2829 or 2850 <= s <= 2899 or 3200 <= s <= 3399: return "Materials"
    if 5000 <= s <= 5999 or 3710 <= s <= 3716 or 2200 <= s <= 2399 or 2500 <= s <= 2599 or 3140 <= s <= 3149 \
            or 7000 <= s <= 7299 or 7900 <= s <= 7999 or 8200 <= s <= 8299 or 1500 <= s <= 1799 and False: return "Consumer Discretionary"
    return "Industrials"
P["sector"] = P.sic.map(sector)

N_PER_SIZE, SEED = 100, 42
rng = np.random.default_rng(SEED)
picked = []
for size, g in P.groupby("size_bucket", observed=True):
    share = g.sector.value_counts() / len(g) * N_PER_SIZE
    alloc = np.floor(share).astype(int)
    for s in (share - alloc).sort_values(ascending=False).index[: N_PER_SIZE - alloc.sum()]:
        alloc[s] += 1                                          # largest-remainder rounding
    for s, k in alloc.items():
        cand = g[g.sector == s].sort_values("symbol_2016")
        picked.append(cand.iloc[rng.choice(len(cand), size=k, replace=False)])
S = pd.concat(picked)

# full-period history (asof 2016 entity) -> first/last trading day, delisting flag
H = {"APCA-API-KEY-ID": data_fetch.get_key("ALPACA_API_KEY_ID"), "APCA-API-SECRET-KEY": data_fetch.get_key("ALPACA_API_SECRET_KEY")}
last, nbars = {}, {}
syms = S.symbol_2016.tolist()
for i in range(0, len(syms), 50):
    params = {"symbols": ",".join(syms[i:i+50]), "timeframe": "1Day", "start": "2016-01-01T00:00:00Z",
              "end": "2024-12-31T23:59:59Z", "limit": 10000, "feed": "sip", "adjustment": "all", "asof": "2016-01-04"}
    while True:
        for k in range(6):
            r = requests.get("https://data.alpaca.markets/v2/stocks/bars", headers=H, params=params, timeout=120)
            if r.ok: break
            time.sleep(2 * 2 ** k)
        js = r.json()
        for sym, bars in (js.get("bars") or {}).items():
            if bars:
                last[sym] = max(last.get(sym, ""), bars[-1]["t"][:10]); nbars[sym] = nbars.get(sym, 0) + len(bars)
        if not js.get("next_page_token"): break
        params["page_token"] = js["next_page_token"]
S["last_bar"] = S.symbol_2016.map(last); S["n_bars_2016_2024"] = S.symbol_2016.map(nbars)
S["delisted_before_2025"] = S.last_bar < "2024-12-01"
S["link_method"] = np.select([S.confirmed_cik, S.name_score >= 0.5], ["SEC ticker map", "name match"], "float/price check")
S.to_parquet(f"{sp}/selected.parquet")
P.to_parquet(f"{sp}/eligible.parquet")
print(pd.crosstab(S.sector, S.size_bucket, margins=True))
print("\ndelisted before 2025:", int(S.delisted_before_2025.sum()), "of", len(S))
print(S.link_method.value_counts().to_dict())
