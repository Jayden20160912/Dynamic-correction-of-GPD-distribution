import io, sys, zipfile
sys.path.insert(0, "/Users/jdzhou/Documents/TQT research project")
import data_fetch
raw = sys.argv[1]
for q in ["2015q4", "2016q1"]:
    r = data_fetch._sec_get(f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        open(f"{raw}/sub_{q}.txt", "wb").write(z.read("sub.txt"))
    print(q, len(r.content))
