"""Interpolate compatible frozen encoder checkpoints; never fits diagnostic labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interpolate(public, adapted, alpha, device):
    import torch
    if alpha not in (0., .25, .5, .75, 1.):
        raise ValueError("Unsupported predeclared alpha")
    if public.keys() != adapted.keys():
        raise ValueError("Checkpoint keys differ")
    output = {}
    with torch.no_grad():
        for name, first in public.items():
            second = adapted[name]
            if first.shape != second.shape or first.dtype != second.dtype:
                raise ValueError(f"Checkpoint tensor shape/dtype differs: {name}")
            if not first.is_floating_point():
                if not torch.equal(first, second):
                    raise ValueError(f"Nonmatching integer checkpoint tensor: {name}")
                output[name] = first.clone().cpu()
                continue
            if not torch.isfinite(first).all() or not torch.isfinite(second).all():
                raise ValueError(f"Checkpoint must be finite: {name}")
            if alpha in (0., 1.):
                output[name] = (first if alpha == 0 else second).clone().cpu()
            else:
                output[name] = torch.lerp(first.to(device), second.to(device), alpha).cpu()
            if not torch.isfinite(output[name]).all():
                raise ValueError(f"Interpolation must be finite: {name}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public", "adapted", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("public-sha", "adapted-sha"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--alpha", type=float, choices=(0., .25, .5, .75, 1.), required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Preserve existing interpolation evidence")
    if digest(args.public) != args.public_sha or digest(args.adapted) != args.adapted_sha:
        raise ValueError("Endpoint checkpoint digest mismatch")
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise ValueError("Use an available GPU for interpolation")
    started = time.monotonic()
    public = torch.load(args.public, map_location="cpu", weights_only=True)
    adapted = torch.load(args.adapted, map_location="cpu", weights_only=True)
    state = interpolate(public, adapted, args.alpha, "cuda:0")
    with args.output.open("xb") as stream:
        torch.save(state, stream)
    record = {"method": "weight-interpolation", "alpha": args.alpha,
              "formula": "(1-alpha)*public-dinov2 + alpha*adapted-simsiam baseline-current-recipe seed42",
              "public_sha256": args.public_sha, "adapted_sha256": args.adapted_sha,
              "checkpoint_sha256": digest(args.output), "seconds": time.monotonic() - started,
              "encoder_fit": "No training or diagnostic gradients; interpolation only"}
    with args.output.with_suffix(".json").open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
