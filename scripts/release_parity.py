"""Parity of one packaged model: torch == transformers == ONNX, and against final_eval.json.

Runs on the pod after build_release_package.py (CPU is enough). Every val
image goes through the release preprocessing (numpy/PIL, checked against the
torchvision transform on the first images) and through the three forms plus
the fp16 ONNX copy. Every gate is hard (registered in dermssl.release.package):

* the package was built with ``--device cuda``, whose fp16-autocast features
  are the ones final_eval.json was scored on;
* the refit evaluation path and the released fused head, on those features,
  give exactly the committed predictions and ``results[role].macro_f1`` (1e-9);
* on the first ``--n-images`` val images the hub fp32 path, torch,
  transformers and ONNX agree within atol 1e-4 with identical argmax, and the
  three forms give the same val macro-F1.

``deployment_metrics`` is reported, not gated: ONNX fp32 on CPU end to end
(own preprocessing, fp32 features) over val, with each metric's delta to the
evaluated value and the predictions that changed; the fp16 copy gets the same
block plus its registered publication gate.

    python scripts/release_parity.py --ham-root ~/work/datos/ham10000 \\
        --model-dir release/v7/B_s42 --final-eval resultados/clasificacion/v7/final_eval.json
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

from dermssl.release import package  # noqa: E402

FORMS = ("torch", "transformers", "onnx")
AUTOCAST_NOTE = ("final_eval.json was scored on features extracted under CUDA fp16 autocast "
                 "(cluster_frozen_bank.extract_features); the release forms run in fp32.")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ham-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=None, help="default release/v7_work/<model>")
    parser.add_argument("--final-eval", type=Path, required=True)
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args(argv)


def collect(runner, reference_fn, batches, n_ref: int) -> tuple[dict, np.ndarray]:
    """Every form over every batch, and the hub reference over the first ``n_ref`` images."""
    forms, reference, seen = {}, [], 0
    for batch in batches:
        for name, proba in runner.predict(batch).items():
            forms.setdefault(name, []).append(proba)
        if seen < n_ref:
            reference.append(reference_fn(batch[: n_ref - seen]))
            seen += len(reference[-1])
    return {k: np.concatenate(v) for k, v in forms.items()}, np.concatenate(reference)


def _matches_json(y_val, proba, json_result: dict, npz_proba) -> dict:
    """Exact reproduction: same predictions as the committed ones and the JSON macro-F1."""
    record = package.f1_against_json(y_val, proba, json_result["macro_f1"], npz_proba)
    record["equal"] = record["equal"] and record["argmax_changes_vs_reference"] == 0
    return record


def deployment_block(y_val, forms: dict, json_result: dict, npz_proba) -> dict:
    """ONNX fp32 (and fp16 with its gate) end to end on CPU, against the evaluated values."""
    block = {"onnx_fp32": package.deployment_metrics(y_val, forms["onnx"], json_result, npz_proba)}
    if "onnx_fp16" in forms:
        block["onnx_fp16"] = {
            **package.deployment_metrics(y_val, forms["onnx_fp16"], json_result, npz_proba),
            "gate": package.fp16_rule(y_val, forms["torch"], forms["onnx_fp16"])}
    return block


def parity_report(forms: dict, reference: np.ndarray, y_val, json_result: dict, npz_proba,
                  eval_proba, head_proba, packaged_on_cuda: bool) -> dict:
    n_ref = len(reference)
    agree = package.forms_agree({"hub_fp32": reference, **{f: forms[f][:n_ref] for f in FORMS}})
    f1s = [package.f1_against_json(y_val, forms[f], json_result["macro_f1"])["macro_f1"] for f in FORMS]
    evaluation = _matches_json(y_val, eval_proba, json_result, npz_proba)
    evaluation["max_abs_diff_vs_committed_probas"] = float(np.abs(eval_proba - npz_proba).max())
    head = _matches_json(y_val, head_proba, json_result, npz_proba)
    gates = {"packaged_on_cuda": bool(packaged_on_cuda),
             "evaluation_path_equals_json": evaluation["equal"],
             "packaged_head_equals_json": head["equal"],
             "forms_agree_first_images": agree["pass"],
             "forms_same_val_macro_f1": max(f1s) - min(f1s) <= package.JSON_F1_TOLERANCE}
    return {"n_images_compared": n_ref, "n_val": int(len(y_val)), "forms_agree": agree,
            "forms_val_macro_f1": dict(zip(FORMS, f1s)), "evaluation_path": evaluation,
            "packaged_head": head, "gates": gates, "pass": all(gates.values()),
            "deployment_metrics": deployment_block(y_val, forms, json_result, npz_proba)}


def val_batches(val_ids, index, batch_size: int):
    from PIL import Image  # noqa: PLC0415

    from dermssl.release import preprocess  # noqa: PLC0415

    for start in range(0, len(val_ids), batch_size):
        arrays = []
        for image_id in val_ids[start:start + batch_size]:
            with Image.open(index[image_id]) as image:
                arrays.append(preprocess.preprocess_image(image))
        yield np.stack(arrays)


def preprocess_check(val_ids, index, n: int) -> float:
    """Max |numpy preprocessing - torchvision eval_transform| on the first ``n`` val images."""
    import cluster_frozen_bank as bank  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from dermssl.release import preprocess  # noqa: PLC0415

    transform, worst = bank.eval_transform(224), 0.0
    for image_id in val_ids[:n]:
        with Image.open(index[image_id]) as image:
            rgb = image.convert("RGB")
            diff = np.abs(preprocess.preprocess_image(rgb) - transform(rgb).numpy()).max()
        worst = max(worst, float(diff))
    return worst


def _inputs(args, release: dict):
    import cluster_frozen_bank as bank  # noqa: PLC0415
    from c1_build_val_index import rebuild_split  # noqa: PLC0415

    from dermssl.feature_cache import fingerprint_ids  # noqa: PLC0415

    split, _ = rebuild_split(args.ham_root.expanduser())
    val_ids = list(split["val_ids"])
    if fingerprint_ids(val_ids) != release["val_fingerprint"]:
        raise SystemExit("ABORT: val ids differ from the ones the package was built with")
    return val_ids, bank.index_images([args.ham_root.expanduser()])


def main(argv=None) -> int:
    import cluster_frozen_bank as bank  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from dermssl.release import torch_forms  # noqa: PLC0415

    args = parse_args(argv)
    release = json.loads((args.model_dir / "release.json").read_text(encoding="utf-8"))
    doc = json.loads(args.final_eval.read_text(encoding="utf-8"))
    work = args.work or REPO / "release" / "v7_work" / release["model"]
    with np.load(work / "eval.npz") as npz:
        y_val, eval_proba, head_proba, weight, bias = (
            npz[k] for k in ("y_val", "eval_proba", "head_proba", "weight", "bias"))
    with np.load(args.final_eval.with_suffix(".probas.npz")) as committed:
        npz_proba = committed[release["role"]]
    val_ids, index = _inputs(args, release)
    hub = bank.build_frozen_backbone(release["backbone_arch"], release["source_weights"], 224,
                                     torch.device(args.device))["model"]
    reference_fn = lambda b: torch_forms.hub_probas(hub, weight, bias, b, args.device)  # noqa: E731
    runner = torch_forms.FormRunner(args.model_dir, device=args.device)
    forms, reference = collect(runner, reference_fn, val_batches(val_ids, index, args.batch_size),
                               args.n_images)
    report = parity_report(forms, reference, y_val, doc["results"][release["role"]], npz_proba,
                           eval_proba, head_proba, release["cuda_fp16_autocast"])
    report.update(model=release["model"], device=args.device, note=AUTOCAST_NOTE,
                  build_device=release["device"],
                  preprocess_max_abs_diff=preprocess_check(val_ids, index, args.n_images))
    package.write_json(args.model_dir / "parity.json", report)
    fp16 = report["deployment_metrics"].get("onnx_fp16", {}).get("gate", {}).get("publish", False)
    print(json.dumps(report["gates"], indent=1), f"\nfp16 publish: {fp16}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
