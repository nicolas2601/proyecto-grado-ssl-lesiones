"""Package one v7 backbone as a release model: refit the probe, fuse it, write the forms.

Runs on the pod, where the weights live. The probe is refit EXACTLY as
``scripts/c1_final_eval.py --split val`` does: same C1 fold-0 train ids (the
train fingerprint is checked), same bank extraction
(``cluster_frozen_bank.extract_features``: Resize 255 + CenterCrop 224 +
ImageNet norm, CUDA fp16 autocast when the device is cuda), same ``logreg``
settings (``probes.fit_logreg``). The fused head and the backbone become
``hf/`` (Dinov2Model + head) and ``onnx/`` (fp32 and fp16 end to end).

    V=/home/jovyan/work/proyectonp/salidas/v7
    python scripts/build_release_package.py --ham-root ~/work/datos/ham10000 \\
        --model B_s42 --weights $V/final_B/simsiam_dinov2_backbone_final.pt \\
        --final-eval resultados/clasificacion/v7/final_eval.json --device cuda

``--device cuda`` reproduces the evaluation's fp16-autocast features, so the
refit probe is the evaluated one; on cpu the features are fp32 and the probe
differs slightly (release_parity.py reports by how much).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dermssl import c1_split_identity as si  # noqa: E402
from dermssl import final_eval as fe  # noqa: E402
from dermssl import probes  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl.feature_cache import fingerprint_ids  # noqa: E402
from dermssl.release import fuse, package, preprocess  # noqa: E402

ARCH = "dinov2_vitb14"
RESOLUTION = 224
ARMS = {"A": ("plain", "A*"), "B": ("proposed", "B*")}
PROBE_SETTINGS = {"probe": "logreg", "scaler": "StandardScaler()", "max_iter": 2000, "C": 1.0,
                  "random_state": 42, "class_weight": None, "solver": "lbfgs (sklearn default)"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ham-root", type=Path, required=True)
    parser.add_argument("--model", required=True, help="<A|B>_s<seed>, e.g. B_s42")
    parser.add_argument("--weights", type=Path, required=True, help="the evaluated backbone .pt")
    parser.add_argument("--final-eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=REPO / "release" / "v7")
    parser.add_argument("--work", type=Path, default=REPO / "release" / "v7_work")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args(argv)


def model_identity(name: str) -> dict:
    arm, _, seed = name.partition("_s")
    if arm not in ARMS or not seed.isdigit():
        raise ValueError(f"model {name!r} is not <A|B>_s<seed>")
    role, label = ARMS[arm]
    return {"model": name, "arm": arm, "arm_label": label, "role": role, "seed": int(seed)}


def check_evaluated_spec(doc: dict, role: str, weights: Path) -> None:
    """The weights packaged must be the ones final_eval.json scored."""
    if doc.get("split") != "val":
        raise ValueError("final_eval.json is not the val evaluation")
    evaluated = doc["specs"][role].split(":", 1)[1]
    if Path(evaluated).expanduser().resolve() != weights.expanduser().resolve():
        raise ValueError(f"{role} was evaluated from {evaluated}, not {weights}")


def check_split(split: dict, doc: dict) -> dict:
    fe.check_val_protocol(split["train_ids"], split["val_ids"])
    if si.ids_sha256(split["val_ids"]) != doc["eval_ids_sha256"]:
        raise ValueError("val ids differ from the ones final_eval.json was scored on")
    return {"train_fingerprint": fingerprint_ids(list(split["train_ids"])),
            "val_fingerprint": fingerprint_ids(list(split["val_ids"])),
            "eval_ids_sha256": doc["eval_ids_sha256"],
            "n_train": len(split["train_ids"]), "n_val": len(split["val_ids"])}


def refit_and_fuse(x_tr, y_tr, x_va, n_classes: int) -> dict:
    scaler, logreg = probes.fit_logreg(x_tr, y_tr)
    weight, bias = fuse.fuse_probe(scaler, logreg, n_classes=n_classes)
    eval_proba = logreg.predict_proba(scaler.transform(x_va))
    fused = fuse.fused_predict_proba(x_va, weight, bias)
    return {"weight": weight, "bias": bias, "eval_proba": eval_proba,
            "fused_max_abs_diff": float(np.abs(fused - eval_proba).max())}


def package_model(hub_model, features: dict, classes: list, out_dir: Path, work_dir: Path,
                  meta: dict, fp16: bool = True) -> dict:
    """Refit + fuse + forms + release.json + the evaluation probabilities for parity."""
    from dermssl.release import torch_forms  # noqa: PLC0415

    probe = refit_and_fuse(features["x_tr"], features["y_tr"], features["x_va"], len(classes))
    torch_forms.write_model_forms(hub_model.cpu(), probe["weight"], probe["bias"], classes,
                                  out_dir, preprocess.PREPROCESSING, fp16=fp16)
    release = {**meta, "classes": classes, "preprocessing": preprocess.PREPROCESSING,
               "probe_settings": PROBE_SETTINGS, "fused_max_abs_diff": probe["fused_max_abs_diff"],
               "backbone_arch": ARCH, "resolution": RESOLUTION}
    package.write_json(out_dir / "release.json", release)
    work_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(work_dir / "eval.npz", y_val=features["y_va"], eval_proba=probe["eval_proba"],
                        weight=probe["weight"], bias=probe["bias"])
    return release


def _extract(args, split, weights: Path):
    import cluster_frozen_bank as bank  # noqa: PLC0415
    import torch  # noqa: PLC0415

    device = torch.device(args.device)
    model = bank.build_frozen_backbone(ARCH, str(weights), RESOLUTION, device)["model"]
    index = bank.index_images([args.ham_root.expanduser()])
    run = lambda ids: bank.extract_features(model, list(ids), index, split["id2label"],  # noqa: E731
                                            RESOLUTION, device, args.batch_size, args.num_workers)
    x_tr, y_tr = run(split["train_ids"])
    x_va, y_va = run(split["val_ids"])
    return model, {"x_tr": x_tr, "y_tr": y_tr, "x_va": x_va, "y_va": y_va}, device


def _meta(args, ident, doc, split_meta, weights: Path, device) -> dict:
    run_json = weights.parent / "run.json"
    run = json.loads(run_json.read_text(encoding="utf-8")) if run_json.is_file() else {}
    return {**ident, **split_meta, "code_revision": rp.clean_revision(REPO),
            "final_eval": str(args.final_eval), "final_eval_sha256": package.sha256_file(args.final_eval),
            "final_eval_code_revision": doc["code_revision"],
            "json_macro_f1": doc["results"][ident["role"]]["macro_f1"],
            "source_weights": str(weights), "source_weights_sha256": package.sha256_file(weights),
            "source_run": {k: run.get(k) for k in ("code_revision", "exclude_ids_sha256", "seed")},
            "device": str(device), "cuda_fp16_autocast": device.type == "cuda"}


def main(argv=None) -> int:
    from c1_build_val_index import rebuild_split  # noqa: PLC0415

    args = parse_args(argv)
    ident = model_identity(args.model)
    weights = args.weights.expanduser()
    doc = json.loads(args.final_eval.read_text(encoding="utf-8"))
    check_evaluated_spec(doc, ident["role"], weights)
    split, _ = rebuild_split(args.ham_root.expanduser())
    split_meta = check_split(split, doc)
    rp.clean_revision(REPO)  # refuse a dirty tree before hours of extraction
    model, features, device = _extract(args, split, weights)
    meta = _meta(args, ident, doc, split_meta, weights, device)
    out_dir, work_dir = args.out / args.model, args.work / args.model
    package_model(model, features, list(split["classes"]), out_dir, work_dir, meta, not args.no_fp16)
    for row in package.size_report(out_dir):
        print(f"{row['mib']:>10.2f} MiB  {row['path']}")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
