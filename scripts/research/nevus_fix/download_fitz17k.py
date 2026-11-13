import csv, json, os, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
MAP = {"melanoma": "MEL", "malignant melanoma": "MEL", "superficial spreading melanoma ssm": "MEL", "lentigo maligna": "MEL",
       "nevocytic nevus": "NEV", "congenital nevus": "NEV", "halo nevus": "NEV",
       "basal cell carcinoma": "BCC", "basal cell carcinoma morpheiform": "BCC", "solid cystic basal cell carcinoma": "BCC",
       "squamous cell carcinoma": "SCC", "actinic keratosis": "ACK", "seborrheic keratosis": "SEK",
       "dermatofibroma": "DF", "pyogenic granuloma": "VASC", "granuloma pyogenic": "VASC"}
rows = [r for r in csv.DictReader(open("fitzpatrick17k.csv"))
        if r["label"] in MAP and "wrong" not in r["qc"].lower() and "atlasdermatologico" in r["url"]]
os.makedirs("images", exist_ok=True)


def get(r):
    out = "images/%s.jpg" % r["md5hash"]
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return r, out, "cached"
    try:
        req = urllib.request.Request(r["url"], headers={"User-Agent": "Mozilla/5.0 (derm-eval-research)"})
        data = urllib.request.urlopen(req, timeout=30).read()
        if len(data) < 1000:
            return r, None, "small"
        open(out, "wb").write(data)
        return r, out, "ok"
    except Exception as e:  # noqa: BLE001
        return r, None, type(e).__name__


res = list(ThreadPoolExecutor(16).map(get, rows))
print(len(rows), Counter(s for _, _, s in res))
with open("fitz_rows.jsonl", "w") as f:
    for r, out, s in res:
        if out:
            f.write(json.dumps({"image_id": r["md5hash"], "image_path": os.path.abspath(out), "label": MAP[r["label"]],
                                "raw_label": r["label"], "fitzpatrick": r["fitzpatrick_scale"], "qc": r["qc"],
                                "url": r["url"]}) + "\n")
