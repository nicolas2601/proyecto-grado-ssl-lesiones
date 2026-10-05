"""onnxruntime inference and the fp16 copy of the exported classifier (no torch)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

INPUT_NAME = "pixel_values"
OUTPUT_NAME = "probabilities"
CPU = ("CPUExecutionProvider",)


def session(path, providers=CPU):
    import onnxruntime as ort  # noqa: PLC0415

    return ort.InferenceSession(str(path), providers=list(providers))


def predict(sess, batch: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Probabilities in float64 for a float32 ``(N, 3, 224, 224)`` batch."""
    outputs = []
    for start in range(0, len(batch), batch_size):
        chunk = np.ascontiguousarray(batch[start:start + batch_size], dtype=np.float32)
        outputs.append(sess.run([OUTPUT_NAME], {INPUT_NAME: chunk})[0])
    return np.concatenate(outputs).astype(np.float64)


def convert_fp16(src, dst) -> Path:
    """fp16 weights and compute, float32 input and output (``keep_io_types``)."""
    import onnx  # noqa: PLC0415
    from onnxconverter_common import float16  # noqa: PLC0415

    model = float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
    onnx.save(model, str(dst))
    return Path(dst)
