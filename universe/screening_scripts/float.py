import sys
sys.path.insert(0, "/Users/jdzhou/Documents/TQT research project")
import data_fetch
raw = sys.argv[1]
for q in ["CY2014Q4I", "CY2015Q1I", "CY2015Q2I", "CY2015Q3I", "CY2015Q4I"]:
    r = data_fetch._sec_get(f"https://data.sec.gov/api/xbrl/frames/dei/EntityPublicFloat/USD/{q}.json")
    open(f"{raw}/float_{q}.json", "wb").write(r.content); print(q, len(r.content))
