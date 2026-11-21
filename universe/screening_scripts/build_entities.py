import json, re, sys
import numpy as np, pandas as pd
raw = sys.argv[1]; out = sys.argv[2]

# ---- SEC filers (10-K/10-Q, 2015Q4-2016Q1): CIK, name, SIC, 2016 ticker (XBRL instance prefix)
s = pd.concat([pd.read_csv(f"{raw}/sub_{q}.txt", sep="\t", dtype=str) for q in ["2015q4", "2016q1"]])
s = s[s.form.isin(["10-K", "10-Q", "10-K/A", "10-Q/A"])].copy()
s["prefix"] = s.instance.str.split("-").str[0].str.upper()
s = s.sort_values("filed")
firm = s.groupby("cik").agg(name=("name", "last"), former=("former", "last"), sic=("sic", "last"),
                            country=("countryba", "last"), prefix=("prefix", "last"),
                            prefixes=("prefix", lambda x: sorted(set(x)))).reset_index()

# ---- Jan-2016 bars keyed by the 2016 symbol (asof=2016-01-04)
b16 = pd.concat([pd.read_parquet(f"{raw}/jan2016_bars.parquet"), pd.read_parquet(f"{raw}/jan2016_bars_prefix.parquet")])
b16 = b16.drop_duplicates(["symbol", "date"])
bcur = pd.read_parquet(f"{raw}/jan2016_bars_current_symbol.parquet")
def stats(b):
    b = b.assign(dv=b.close * b.volume)
    g = b.groupby("symbol")
    st = pd.DataFrame({"n_days": g.size(), "px": g.close.median(), "dollar_vol": g.dv.median()})
    fp = b.sort_values("date").groupby("symbol").apply(lambda d: hash(tuple(zip(d.date, d.close.round(4), d.volume))), include_groups=False)
    return st.assign(fp=fp)
st16, stcur = stats(b16), stats(bcur)

# ---- Alpaca names (current/last ticker of each entity) and SEC current ticker map
assets = pd.DataFrame(json.load(open(f"{raw}/assets_active.json")) + json.load(open(f"{raw}/assets_inactive.json")))
names_by_sym = assets.groupby("symbol").name.apply(list).to_dict()
cur_names_by_fp = {}
for sym, fp in stcur.fp.items():
    cur_names_by_fp.setdefault(fp, []).extend([(sym, n) for n in names_by_sym.get(sym, [])])
ct = json.load(open(f"{raw}/company_tickers.json"))
cur_tickers = {}
for r in ct.values():
    cur_tickers.setdefault(str(r["cik_str"]), []).append(r["ticker"].upper())

STOP = {"INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC", "LLC", "LP", "THE", "NEW",
        "HOLDINGS", "HOLDING", "GROUP", "COMMON", "STOCK", "SHARES", "CLASS", "A", "B", "C", "ORDINARY",
        "TRUST", "OF", "AND", "DE", "NV", "MD", "SA", "NA", "INCORPORATED", "INTERNATIONAL", "INTL", "PAR", "VALUE"}
def toks(n):
    n = re.sub(r"/[A-Z]{2,3}/?", " ", str(n).upper())
    return {t for t in re.split(r"[^A-Z0-9]+", n) if t and t not in STOP}
def sim(a, b):
    ta, tb = toks(a), toks(b)
    return len(ta & tb) / max(1, min(len(ta), len(tb))) if ta and tb else 0.0

rows = []
for f in firm.itertuples():
    for p in f.prefixes[::-1]:                       # latest prefix first
        if p not in st16.index: continue
        fp = st16.at[p, "fp"]
        via_cik = any(t in stcur.index and stcur.at[t, "fp"] == fp for t in cur_tickers.get(f.cik, []))
        alp_names = [n for _, n in cur_names_by_fp.get(fp, [])] + names_by_sym.get(p, [])
        name_score = max([sim(f.name, n) for n in alp_names] + [sim(f.former, n) for n in alp_names if isinstance(f.former, str)] + [0])
        rows.append(dict(cik=f.cik, name=f.name, sic=f.sic, country=f.country, symbol_2016=p,
                         confirmed_cik=via_cik, name_score=name_score, alpaca_name=(alp_names or [""])[0],
                         current_symbols=",".join(sorted({s for s, _ in cur_names_by_fp.get(fp, [])})),
                         **st16.loc[p, ["n_days", "px", "dollar_vol"]].to_dict()))
        break
E = pd.DataFrame(rows)
E["linked"] = E.confirmed_cik | (E.name_score >= 0.5)
# one CIK per 2016 symbol: drop ambiguous duplicates
E = E.sort_values(["linked", "confirmed_cik", "name_score"], ascending=False)
E["dup"] = E.duplicated("symbol_2016")
print("SEC filers:", len(firm), "| with Jan-2016 Alpaca bars:", len(E),
      "| link confirmed (CIK map):", int(E.confirmed_cik.sum()), "| by name:", int((~E.confirmed_cik & E.linked).sum()),
      "| unlinked:", int((~E.linked).sum()), "| duplicate symbols:", int(E.dup.sum()))

# ---- shares outstanding around 2015 year-end (dei cover page) -> market cap
sh = []
for q in ["CY2015Q4I", "CY2015Q3I", "CY2016Q1I"]:
    d = pd.DataFrame(json.load(open(f"{raw}/frames_{q}.json"))["data"]); d["q"] = q; sh.append(d)
sh = pd.concat(sh); sh = sh[sh.end <= "2016-03-31"]
sh["cik"] = sh.cik.astype(str)
sh = sh.sort_values(["cik", "end"]).groupby("cik").last()
E["shares"] = E.cik.map(sh.val); E["shares_date"] = E.cik.map(sh.end)
E["mktcap"] = E.shares * E.px
E.to_parquet(f"{out}/entities.parquet")
print("with shares outstanding:", int(E.shares.notna().sum()))
print(E[E.symbol_2016.isin(["TWTR","FB","BBBY","SIVB","SHLD","AAPL","CELG","FRC","GE","XLNX"])][
    ["symbol_2016","name","sic","linked","confirmed_cik","name_score","current_symbols","mktcap"]].to_string())
