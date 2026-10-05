"""Release bookkeeping without torch: head file, checksums, sizes, parity gates.

The gates are registered here, beside their tests, so the pod run computes
its verdicts instead of someone reading them off a table:

* forms agree: every pair of forms within ``atol`` 1e-4 and identical argmax;
* fp16 copy: published only if its argmax matches torch fp32 on >= 99.5 % of
  val AND ``|delta macro-F1| <= 0.002``; otherwise the release is fp32 only.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from dermssl import metrics

GITHUB_ASSET_LIMIT = 2 * 1024**3
FORMS_ATOL = 1e-4
JSON_F1_TOLERANCE = 1e-9
FP16_MIN_AGREEMENT = 0.995
FP16_MAX_DELTA_F1 = 0.002


class AssetTooLarge(ValueError):
    """A file exceeds the per-asset limit of a GitHub release."""


def write_head(path, weight, bias, classes, preprocessing: dict) -> Path:
    """Fused head as safetensors: ``weight`` (C, D) and ``bias`` (C,) float32 + metadata."""
    weight = np.ascontiguousarray(weight, dtype=np.float32)
    bias = np.ascontiguousarray(bias, dtype=np.float32)
    if weight.shape[0] != len(classes) or bias.shape != (len(classes),):
        raise ValueError(f"head has {weight.shape[0]} rows for {len(classes)} classes")
    from safetensors.numpy import save_file  # noqa: PLC0415

    metadata = {"format": "pt", "classes": json.dumps(list(classes)),
                "preprocessing": json.dumps(preprocessing)}
    save_file({"weight": weight, "bias": bias}, str(path), metadata=metadata)
    return Path(path)


def read_head(path) -> dict:
    from safetensors import safe_open  # noqa: PLC0415

    with safe_open(str(path), framework="numpy") as head:
        meta = head.metadata()
        return {"weight": head.get_tensor("weight"), "bias": head.get_tensor("bias"),
                "classes": json.loads(meta["classes"]),
                "preprocessing": json.loads(meta["preprocessing"])}


def sha256_file(path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path, skip: tuple[str, ...] = ()) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name not in skip)


def write_sha256sums(root, name: str = "SHA256SUMS") -> Path:
    """``sha256sum``-compatible list of every file under ``root`` (itself excluded)."""
    root = Path(root)
    lines = [f"{sha256_file(p)}  {p.relative_to(root).as_posix()}" for p in _files(root, (name,))]
    out = root / name
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def size_report(root, limit_bytes: int = GITHUB_ASSET_LIMIT) -> list[dict]:
    """Every file, largest first; raises if one is over the per-asset limit."""
    root = Path(root)
    report = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
               "mib": round(p.stat().st_size / 2**20, 2)} for p in _files(root)]
    report.sort(key=lambda r: (-r["bytes"], r["path"]))
    over = [r["path"] for r in report if r["bytes"] > limit_bytes]
    if over:
        raise AssetTooLarge(f"over {limit_bytes} bytes: {', '.join(over)}")
    return report


def forms_agree(probas: dict, atol: float = FORMS_ATOL) -> dict:
    """Pairwise max |delta p| and argmax mismatches between forms on the same images."""
    pairs = {}
    for (a, pa), (b, pb) in itertools.combinations(probas.items(), 2):
        mismatches = int((pa.argmax(1) != pb.argmax(1)).sum())
        max_diff = float(np.abs(pa - pb).max())
        pairs[f"{a}-{b}"] = {"max_abs_diff": max_diff, "argmax_mismatches": mismatches,
                             "argmax_identical": mismatches == 0,
                             "pass": mismatches == 0 and max_diff <= atol}
    return {"atol": atol, "pairs": pairs, "pass": all(p["pass"] for p in pairs.values())}


def f1_against_json(y_true, proba, json_f1: float, reference_proba=None,
                    tolerance: float = JSON_F1_TOLERANCE, n_classes: int = 7) -> dict:
    """Macro-F1 of ``proba`` against the committed evaluation value."""
    f1 = float(metrics.summarize(y_true, proba, n_classes)["macro_f1"])
    out = {"macro_f1": f1, "json_macro_f1": float(json_f1), "delta": f1 - float(json_f1),
           "tolerance": tolerance, "equal": abs(f1 - float(json_f1)) <= tolerance}
    if reference_proba is not None:
        out["argmax_changes_vs_reference"] = int((proba.argmax(1) != reference_proba.argmax(1)).sum())
    return out


def fp16_rule(y_true, fp32_proba, fp16_proba, n_classes: int = 7) -> dict:
    """Registered rule for publishing the fp16 ONNX copy."""
    agreement = float((fp32_proba.argmax(1) == fp16_proba.argmax(1)).mean())
    f1_32 = float(metrics.summarize(y_true, fp32_proba, n_classes)["macro_f1"])
    f1_16 = float(metrics.summarize(y_true, fp16_proba, n_classes)["macro_f1"])
    delta = f1_16 - f1_32
    return {
        "rule": {"min_argmax_agreement": FP16_MIN_AGREEMENT, "max_abs_delta_macro_f1": FP16_MAX_DELTA_F1},
        "argmax_agreement": agreement, "macro_f1_fp32": f1_32, "macro_f1_fp16": f1_16,
        "delta_macro_f1": delta,
        "publish": agreement >= FP16_MIN_AGREEMENT and abs(delta) <= FP16_MAX_DELTA_F1,
    }


JSON_METRICS = ("macro_f1", "balanced_accuracy", "ece")


def _headline(y_true, proba, n_classes: int = 7) -> dict:
    """The six deployment metrics of one probability matrix."""
    from dermssl import per_class_metrics as pcm  # noqa: PLC0415

    summary = metrics.summarize(y_true, proba, n_classes)
    report = pcm.class_report(y_true, proba, pcm.HAM10000_CLASSES)
    return {"macro_f1": summary["macro_f1"], "balanced_accuracy": summary["balanced_accuracy"],
            "accuracy": summary["accuracy"], "mel_sensitivity": report["positive_sensitivity"],
            "mel_auc": report["positive_auc"], "ece": summary["ece"]}


def deployment_metrics(y_true, proba, json_result: dict, evaluated_proba) -> dict:
    """What a user gets from a released form, next to what was evaluated.

    ``evaluated`` is the final_eval.json value where the JSON has it (macro-F1,
    balanced accuracy, ECE) and otherwise is computed from the committed
    evaluation probabilities (accuracy, melanoma sensitivity and AUC).
    """
    deployed = _headline(y_true, proba)
    from_npz = _headline(y_true, evaluated_proba)
    rows = {}
    for name, value in deployed.items():
        in_json = name in JSON_METRICS
        evaluated = float(json_result[name]) if in_json else float(from_npz[name])
        rows[name] = {"value": float(value), "evaluated": evaluated, "delta": float(value) - evaluated,
                      "evaluated_source": "final_eval.json" if in_json else "final_eval.probas.npz"}
    changed = int((np.asarray(proba).argmax(1) != np.asarray(evaluated_proba).argmax(1)).sum())
    return {"n_images": int(len(y_true)), "n_predictions_changed": changed, "metrics": rows}


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path
