"""Corrections found while building the dataset (2026-11-20): the Alpaca ticker follows a
different company after these dates, or the stock was delisted earlier than detected."""
import sys, pandas as pd
d = sys.argv[1]
F = pd.read_csv(f"{d}/selected_universe_300.csv")
fixes = {
    "ACET": ("2019-04-02", "Aceto bankrupt 2019; ticker later reused by Adicet Bio (533-day gap) -> cut"),
    "DLPH": ("2017-12-04", "Delphi Automotive became Aptiv on 2017-12-05; ticker followed the Delphi Technologies spin-off -> cut"),
    "EVHC": ("2016-12-01", "Envision merged with AmSurg on 2016-12-02 (combined firm took the ticker) -> cut"),
    "MDRX": ("2024-02-28", "Allscripts/Veradigm delisted from Nasdaq in Feb 2024"),
}
for t, (end, note) in fixes.items():
    i = F.symbol_2016 == t
    assert i.sum() == 1, t
    F.loc[i, ["history_end", "delisted_before_2025", "history_note"]] = [end, True, note]
F.to_csv(f"{d}/selected_universe_300.csv", index=False)
print(F[F.symbol_2016.isin(fixes)][["symbol_2016", "sec_name", "history_end", "delisted_before_2025"]].to_string(index=False))
print("delisted before 2025:", int(F.delisted_before_2025.sum()))
