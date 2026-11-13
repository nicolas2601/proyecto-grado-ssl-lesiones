#!/usr/bin/env python3
"""End-to-end check through the HTTP API: post each labelled image and tally nevus collapse.

Input: labels.csv (file,label,modality,...) relative to --root. Output: JSONL of responses plus a
summary on stdout. Uses only the public /api/predict contract.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
import uuid
from collections import Counter
from pathlib import Path

HAM7 = {"melanoma": "mel", "bcc": "bcc", "scc": "akiec", "actinic_keratosis": "akiec",
        "seborrheic_keratosis": "bkl", "nevus": "nv", "dermatofibroma": "df", "vascular": "vasc"}
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def post(url: str, path: Path) -> dict:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: {MIME.get(path.suffix.lower(), 'image/jpeg')}\r\n\r\n").encode()
    body += path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}", "User-Agent": "derm-eval-research/0.1"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="https://derm.nicolasmoreno.site")
    parser.add_argument("--model", default="MOCO_HAM7")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pause", type=float, default=1.0)
    args = parser.parse_args()
    url = f"{args.base}/api/predict?model={args.model}&mode=base"
    rows = list(csv.DictReader((args.root / "labels.csv").open(newline="")))
    tally, preds = Counter(), Counter()
    with args.out.open("w") as sink:
        for row in rows:
            truth = HAM7.get(row["label"])
            if truth is None:
                continue
            try:
                answer = post(url, args.root / row["file"])
            except Exception as error:  # noqa: BLE001 - report and continue
                sink.write(json.dumps({"file": row["file"], "error": str(error)}) + "\n")
                continue
            pred = answer.get("pred")
            sink.write(json.dumps({"file": row["file"], "truth": truth, "modality": row["modality"],
                                   "response": answer}) + "\n")
            tally[(truth, pred == truth)] += 1
            preds[pred] += 1
            time.sleep(args.pause)
    n = sum(preds.values())
    correct = sum(v for (t, ok), v in tally.items() if ok)
    non_nv = sum(v for (t, ok), v in tally.items() if t != "nv")
    print(json.dumps({"n": n, "accuracy": correct / max(n, 1), "pred_counts": preds,
                      "pred_nv_rate": preds.get("nv", 0) / max(n, 1), "non_nv_images": non_nv}, default=str))


if __name__ == "__main__":
    main()
