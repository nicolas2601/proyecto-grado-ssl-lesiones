"""Backend SegmenterService (schema v2) vs the package reference predict_image on real images.

  python parity_backend.py <backend_repo> <segmenter_dir> <manifest_sha256> <out.json> <image> [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

repo, seg_dir, pin, out = sys.argv[1:5]
sys.path.insert(0, repo)
sys.path.insert(0, repo + "/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo.backend.segmenter_service import SegmenterError, SegmenterService  # noqa: E402
from unified_seg_preprocess import load_session, predict_image  # noqa: E402

service = SegmenterService(seg_dir, pin)
status = service.status()
assert status["ready"], status
reference = load_session(Path(seg_dir) / "segmenter.onnx")
rows = []
for path in map(Path, sys.argv[5:]):
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    rgb = np.asarray(image)
    expected = predict_image(reference, image)[1]
    try:
        got, reason = service.segment(rgb).full_mask, None
    except SegmenterError as exc:
        got, reason = None, exc.reason
    row = {"image": path.name, "size": list(image.size), "reference_area": round(float(expected.mean()), 4)}
    if got is None:
        row.update({"backend": reason})
    else:
        row.update({"identical": bool(np.array_equal(got, expected)), "differing_px": int((got != expected).sum())})
    rows.append(row)
summary = {"identity": status["identity"], "n": len(rows),
           "identical": sum(r.get("identical", False) for r in rows),
           "backend_reasons": [r for r in rows if "backend" in r], "rows": rows}
Path(out).write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: summary[k] for k in ("n", "identical")}), [r for r in rows if not r.get("identical")][:5])
