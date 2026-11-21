"""Post-sampling fix: drop exchange-traded commodity/index trusts that file 10-Ks (SIC 6799 trusts)
and replace each with a seeded random draw from the same size bucket x sector stratum."""
import sys, numpy as np, pandas as pd
d = sys.argv[1]
F = pd.read_csv(f"{d}/selected_universe_300.csv"); P = pd.read_csv(f"{d}/eligible_pool_2016.csv")
fund = lambda df: (df.sic == 6799) & df.sec_name.str.contains(r"ISHARES|TRUST|INDEX|ETF|FUND", case=False, regex=True)
print("fund-like in pool:", P[fund(P)][["symbol_2016", "sec_name"]].values.tolist())
rng = np.random.default_rng(42)
F["delisted_before_2025"] = F["delisted_before_2025"].astype(object)
for i in F.index[fund(F)]:
    r = F.loc[i]
    cand = P[(P.size_bucket == r.size_bucket) & (P.sector == r.sector) & ~P.symbol_2016.isin(F.symbol_2016) & ~fund(P)]
    new = cand.sort_values("symbol_2016").iloc[rng.integers(len(cand))]
    print(f"replace {r.symbol_2016} ({r.sec_name}) -> {new.symbol_2016} ({new.sec_name})")
    F.loc[i, P.columns.drop("selected")] = new[P.columns.drop("selected")].values
    F.loc[i, ["history_end", "delisted_before_2025", "history_note"]] = [None, None, "replacement draw: history end not yet checked"]
P["selected"] = P.symbol_2016.isin(F.symbol_2016) & ~fund(P)
F.to_csv(f"{d}/selected_universe_300.csv", index=False); P.to_csv(f"{d}/eligible_pool_2016.csv", index=False)
