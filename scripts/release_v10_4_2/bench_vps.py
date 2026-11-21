"""Latency and memory of the DPT segmenter under the derm-api limits."""
import json, resource, sys, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

sys.path.insert(0, "/lab")
from unified_seg_preprocess import postprocess_logits, preprocess_image

model = sys.argv[1]
threads = int(sys.argv[2])
images = [Path(p) for p in sys.argv[3:]]
opts = ort.SessionOptions()
opts.intra_op_num_threads = threads
opts.inter_op_num_threads = 1
t0 = time.perf_counter()
sess = ort.InferenceSession(model, sess_options=opts, providers=["CPUExecutionProvider"])
load_s = time.perf_counter() - t0
rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
times = []
for rep in range(3):
    for p in images:
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        t = time.perf_counter()
        x, meta = preprocess_image(img)
        logits = sess.run(["logits"], {"pixel_values": x})[0]
        probs, mask = postprocess_logits(logits, meta)
        times.append((time.perf_counter() - t) * 1000)
warm = times[1:]
print(json.dumps({"runtime": f"onnxruntime {ort.__version__} CPUExecutionProvider", "model": Path(model).name,
                  "container_limits": {"cpus": 1.5, "memory_mb": 2500}, "threads": {"intra": threads, "inter": 1},
                  "load_s": round(load_s, 1), "rss_after_load_mb": round(rss_after_load),
                  "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                  "first_ms": round(times[0]), "p50_ms": round(float(np.median(warm))),
                  "p90_ms": round(float(np.percentile(warm, 90))), "max_ms": round(max(warm)), "n_timed": len(warm),
                  "scope": "segmenter only: letterbox preprocess + ONNX + softmax/upsample, VPS 2 vCPU host"}))
