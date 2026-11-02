"""CPU-only, synthetic validation of the pinned integration adapters.

This is not a patient-image pilot and never trains a classifier or evaluates
clinical performance. The old OOF probabilities are deliberately not consumed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermssl.pilot_adapters import (  # noqa: E402
    FrozenCLSAdapter, FrozenResNetSegmenter, file_sha256, shift_healthy_skin,
)


def validate(config):
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    # Fixed local process settings; no accelerator initialization or optimizer.
    torch.set_num_threads(4)
    seg = config["segmenter"]
    segmenter = FrozenResNetSegmenter(seg["path"], seg["sha256"], seg["run_id"], device="cpu")
    enc = config["encoder"]
    encoder = FrozenCLSAdapter.from_local_hub(
        enc["hub_directory"], enc["path"], enc["sha256"],
        expected_source_sha256=enc["implementation_sha256"], device="cpu",
    )
    image = np.random.default_rng(20261102).integers(0, 256, (99, 211, 3), dtype=np.uint8)
    reference = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])(Image.fromarray(image))[None]
    preprocess_error = float((reference - encoder.preprocess([image])).abs().max())
    with torch.inference_mode():
        expected = encoder.model.forward_features(reference)["x_norm_clstoken"].cpu()
    actual = encoder.encode([image])
    feature_error = float((actual - expected).abs().max())
    if preprocess_error != 0 or feature_error != 0:
        raise ValueError("Classifier preprocessing/CLS parity differs from the frozen reference")
    mask, qa = segmenter.segment(image)
    shifted, applied = shift_healthy_skin(image, mask, shift_buckets=1)
    preserved = (np.array_equal(shifted[mask.lesion_mask], image[mask.lesion_mask])
                 and np.array_equal(shifted[~mask.valid_fov_mask], image[~mask.valid_fov_mask]))
    if not preserved:
        raise ValueError("Color transformation changed protected pixels")
    features = encoder.encode([image, shifted])
    return {
        "schema_version": 1, "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_validation_only": True, "real_patient_images_used": False,
        "pilot_started": False, "device": "cpu", "rng_seed": 20261102,
        "adapter_source_sha256": file_sha256(Path(__file__).resolve().parents[1] / "src/dermssl/pilot_adapters.py"),
        "validator_source_sha256": file_sha256(__file__),
        "preprocess_max_abs_error_vs_torchvision": preprocess_error,
        "cls_max_abs_error_vs_reference": feature_error,
        "feature_shape": list(features.shape), "finite_features": bool(torch.isfinite(features).all()),
        "segmenter": segmenter.provenance, "encoder": encoder.provenance, "technical_qa": qa,
        "color_transform": {"method": "predicted_skin_Lstar_perturbation", "shift_steps": 1,
                            "requested_Lstar_delta": -8.0, "clinical_fototype_used": False},
        "shift_applied": applied, "lesion_and_outside_fov_pixels_preserved": preserved,
        "reader_evaluated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError("Preserve existing validation evidence; use a new output path")
    result = validate(json.loads(args.config.read_text(encoding="utf-8")))
    result["input_config_sha256"] = file_sha256(args.config)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output_json), "synthetic_validation_only": True}))


if __name__ == "__main__":
    main()
