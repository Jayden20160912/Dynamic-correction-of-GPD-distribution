import sys, pandas as pd
sys.path.insert(0, "/Users/jdzhou/Documents/TQT research project")
import data_fetch
sp = sys.argv[1]
O = pd.read_parquet(f"{sp}/continuity2.parquet"); S = pd.read_parquet(f"{sp}/selected.parquet")
O = O[O.genuine_continuation].merge(S[["symbol_2016", "cik"]], left_on="symbol", right_on="symbol_2016")
rows = []
for r in O.itertuples():
    js = data_fetch._sec_get(f"https://data.sec.gov/submissions/CIK{int(r.cik):010d}.json").json()
    rec = pd.DataFrame(js["filings"]["recent"])
    periodic = rec[rec.form.isin(["10-K", "10-Q"])]
    last_periodic = periodic.filingDate.max() if len(periodic) else "-"
    rows.append((r.symbol, r.name[:28], r.asof_last_bar, r.plain_last_bar, last_periodic, ",".join(js.get("tickers", []))))
R = pd.DataFrame(rows, columns=["symbol", "name", "asof_last_bar", "plain_last_bar", "cik_last_10K_10Q", "sec_current_tickers"])
R["same_company_continues"] = R.cik_last_10K_10Q > R.asof_last_bar
print(R.to_string(index=False)); R.to_parquet(f"{sp}/cik_alive.parquet")
