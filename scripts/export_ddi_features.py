"""Frozen CLS224 features over DDI, using the bank's own extraction.

``export_overlay_features.py`` cannot be reused here: its role whitelist is
``{pad_adapt, ham_dev}`` and it requires the manifest to sit beside an overlay
receipt. So its two primitives - ``load_encoder`` and ``extract`` from the
frozen tree - are reused directly, which is what keeps these features
comparable with the ones already exported for PAD and HAM.

Every image's sha256 is verified against the manifest before it is read, and the
manifest is verified to describe the pinned zip. DDI images never enter this
repository: the Stanford Research Use Agreement is personal, non-commercial and
forbids redistribution. Only features and metrics leave the cluster.

Why DDI matters for this line: it is the only dataset here whose phototype is
assigned by consensus of two board-certified dermatologists rather than
estimated. This project discarded ITA as a measurement instrument
(``docs/63``), so DDI is the only way left to evaluate a fairness claim without
leaning on the instrument that was rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

EXPERIMENT_ROOT = Path("/home/jovyan/work/clf1_clinical_reference_20261024")
BASE = Path("/home/jovyan/work/ddi-20261105")
PINNED_ZIP_SHA = "f9099a51fd451932c5923d2feaf025f2a3e28fae628fc5ae477eb418fe837e9f"


def digest(path: Path) -> str:
    handle = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            handle.update(chunk)
    return handle.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if digest(args.weights) != args.weights_sha:
        raise SystemExit("encoder sha256 mismatch")
    if args.output.exists():
        raise SystemExit("refusing to overwrite existing evidence")

    manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    if manifest["zip_sha256"] != PINNED_ZIP_SHA:
        raise SystemExit("manifest does not describe the pinned zip")

    rows = []
    for record in manifest["images"]:
        image = BASE / "images" / record["file_name"]
        if digest(image) != record["sha256"]:
            raise SystemExit("image sha256 mismatch: " + record["file_name"])
        rows.append({
            "image_id": record["ddi_id"],
            "image_path": str(image),
            "identity_component_id": "ddi-" + record["ddi_id"],
        })

    bindings = json.loads((EXPERIMENT_ROOT / "config/paths.json").read_text())
    frozen = Path(bindings["frozen_experiment_root"])
    sys.path[:0] = [str(EXPERIMENT_ROOT / "scripts"), str(frozen / "scripts"),
                    str(frozen / "code/src")]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from evaluate_final_refit import load_encoder
    from extract_preprocessing_variants import extract

    torch.set_num_threads(2)
    started = time.monotonic()
    model = load_encoder(args.weights)
    features = extract(model, rows, 224, False)
    if features.shape != (len(rows), 768) or not np.isfinite(features).all():
        raise SystemExit("invalid frozen CLS feature matrix")
    np.savez_compressed(args.output, features=features,
                        image_ids=np.asarray([row["image_id"] for row in rows]))
    print(json.dumps({
        "images": len(rows),
        "weights_sha256": args.weights_sha,
        "features_sha256": digest(args.output),
        "every_image_sha_verified": True,
        "seconds": round(time.monotonic() - started, 1),
    }), flush=True)


if __name__ == "__main__":
    main()
