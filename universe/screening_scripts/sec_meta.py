import json, sys
sys.path.insert(0, "/Users/jdzhou/Documents/TQT research project")
import data_fetch
raw = sys.argv[1]
for name, url in [
    ("company_2016q1.idx", "https://www.sec.gov/Archives/edgar/full-index/2016/QTR1/company.idx"),
    ("company_2015q4.idx", "https://www.sec.gov/Archives/edgar/full-index/2015/QTR4/company.idx"),
    ("frames_CY2015Q4I.json", "https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/CY2015Q4I.json"),
    ("frames_CY2016Q1I.json", "https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/CY2016Q1I.json"),
    ("frames_CY2015Q3I.json", "https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/CY2015Q3I.json"),
    ("company_tickers.json", "https://www.sec.gov/files/company_tickers.json"),
]:
    r = data_fetch._sec_get(url)
    open(f"{raw}/{name}", "wb").write(r.content)
    print(name, len(r.content))
