"""Paired ISIC 2018 (2594) evaluation: shipped v10.4.1 segmenter vs the unified DPT candidate.

Both models are scored per image with the frozen thesis evaluator (compute_binary_metrics, HD95
included) against the same validated ground-truth masks.

  python eval_pair.py <out_dir> <shard> <n_shards> <threads>
  python eval_pair.py <out_dir> --aggregate
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SHIPPED = Path("/home/jovyan/work/seg_release_v10_4_1")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SHIPPED))
sys.path.insert(0, str(SHIPPED / "scripts"))
import run_segmenter_onnx as shipped_runner  # noqa: E402
from preprocess import logits_to_mask  # noqa: E402
from segmenter_common import load_eval_index, load_evaluator, load_gt_mask  # noqa: E402
from unified_seg_preprocess import predict_image  # noqa: E402

METRIC_KEYS = ["dice", "iou", "sensitivity", "specificity", "accuracy", "hd95"]
MODELS = ("shipped_v10_4_1", "dpt_s42")
DPT_ONNX = HERE / "vera_unified_dpt_s42_fp32.onnx"
BOOTSTRAP = 2000
SEED = 42


def dpt_session(threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(DPT_ONNX), sess_options=options, providers=["CPUExecutionProvider"])


def run_shard(out_dir: Path, shard: int, n_shards: int, threads: int) -> None:
    index = load_eval_index()
    ids = sorted(index)[shard::n_shards]
    evaluator = load_evaluator()
    shipped = shipped_runner.load_session(SHIPPED / "segmenter.onnx")
    dpt = dpt_session(threads)
    started = time.time()
    with (out_dir / f"cases-{shard}.jsonl").open("w") as cases:
        for n, image_id in enumerate(ids, 1):
            image_path, mask_path = index[image_id]
            gt = load_gt_mask(mask_path)
            with Image.open(image_path) as src:
                width, height = src.size
                preds = {
                    "shipped_v10_4_1": logits_to_mask(shipped_runner.predict_logits(shipped, src), width, height),
                    "dpt_s42": predict_image(dpt, src)[1],
                }
            row = {"image_id": image_id}
            for name, pred in preds.items():
                metrics = evaluator.compute_binary_metrics(np.asarray(pred, dtype=bool), gt, compute_hd=True)
                row[name] = {k: metrics[k] for k in METRIC_KEYS}
            cases.write(json.dumps(row) + "\n")
            if n % 50 == 0:
                print(f"shard {shard}: {n}/{len(ids)} {time.time() - started:.0f}s", flush=True)


def paired_ci(diff: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    means = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(BOOTSTRAP)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def aggregate(out_dir: Path) -> None:
    rows = [json.loads(line) for path in sorted(out_dir.glob("cases-*.jsonl")) for line in path.read_text().splitlines()]
    evaluator = load_evaluator()
    result = {"n_cases": len(rows), "models": {}}
    for name in MODELS:
        agg, ci, den = evaluator.summarize_metric_aggregates({k: [r[name][k] for r in rows] for k in METRIC_KEYS})
        dice = np.asarray([r[name]["dice"] for r in rows])
        result["models"][name] = {"metrics": agg, "bootstrap_ci_95": ci, "denominators": den,
                                  "fail_dice_lt_0_2_pct": float((dice < 0.2).mean() * 100)}
    diff = np.asarray([r["dpt_s42"]["dice"] - r["shipped_v10_4_1"]["dice"] for r in rows])
    result["paired_dice_difference"] = {"mean": float(diff.mean()), "ci95": paired_ci(diff),
                                        "dpt_better_pct": float((diff > 0).mean() * 100)}
    (out_dir / "paired_isic2018_2594.json").write_text(json.dumps(result, indent=2))
    summary = {m: {k: result["models"][m]["metrics"].get(f"mean_{k}") for k in ("dice", "iou", "hd95")} for m in MODELS}
    print(json.dumps({"n": len(rows), **summary, "diff": result["paired_dice_difference"]}, indent=2))


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    if sys.argv[2] == "--aggregate":
        aggregate(out)
    else:
        run_shard(out, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
