#!/usr/bin/env python3
"""Fit the two final development readers; never open sealed evaluation roles.

Defaults reproduce the shipped v10.0.0 readers (C by macro-F1, development rows only). Optional:
--selection-metric balanced_accuracy and --external-* add labelled external rows to the training folds
and the final fit (they never validate). External features must use the exact V10 CLS4@224 recipe.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dermssl.release import moco_head_fit, moco_inputs  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-metric", choices=moco_head_fit.SELECTION_METRICS, default="macro_f1")
    parser.add_argument("--external-manifest", type=Path)
    parser.add_argument("--external-features-dir", type=Path)
    parser.add_argument("--external-status", type=Path)
    args = parser.parse_args()
    external = (args.external_manifest, args.external_features_dir, args.external_status)
    if any(external) and not all(external):
        parser.error("--external-manifest, --external-features-dir and --external-status go together")
    return args


def fit_task(args, task):
    features, labels, groups, _, provenance = moco_inputs.load_development(args.evidence_root, task)
    extra, external_provenance = None, None
    if args.external_manifest is not None:
        ext_x, ext_y, ext_g, external_provenance = moco_inputs.load_external_rows(
            args.external_manifest, args.external_features_dir, args.external_status, task)
        extra = (ext_x, ext_y, ext_g)
    started = time.monotonic()
    with threadpool_limits(limits=2):
        artifact = moco_head_fit.fit_reader(features, labels, groups, moco_inputs.CLASSES[task],
                                            selection_metric=args.selection_metric, extra=extra)
    artifact["provenance"] = provenance
    artifact["external_provenance"] = external_provenance
    artifact["software"] = {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}
    artifact["seconds"] = time.monotonic() - started
    artifact["source_sha256"] = {name: moco_inputs.sha256_file(path) for name, path in (
        ("moco_head_fit.py", Path(moco_head_fit.__file__)), ("moco_inputs.py", Path(moco_inputs.__file__)),
        ("fit_moco_release_readers.py", Path(__file__)))}
    return artifact


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=False)
    for task in ("HAM", "PAD"):
        artifact = fit_task(args, task)
        with (args.output / f"{task}-reader.json").open("x") as stream:
            json.dump(artifact, stream, indent=2, allow_nan=False)
        with (args.output / f"{task}-reader.npz").open("xb") as stream:
            np.savez_compressed(stream, weight=np.asarray(artifact["weight"]), bias=np.asarray(artifact["bias"]),
                                classes=np.asarray(artifact["classes"]))
        print(json.dumps({"task": task, "C": artifact["C"], "n_images": artifact["n_images"],
                          "n_groups": artifact["n_groups"], "extra_images": artifact["extra_images"],
                          "metric": artifact["selection"]["metric"], "seconds": artifact["seconds"],
                          "n_iter": artifact["n_iter"]}), flush=True)


if __name__ == "__main__":
    main()
