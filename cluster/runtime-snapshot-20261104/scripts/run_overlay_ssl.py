"""Explicit isolated adapters for the frozen SSL loop; canonical sources stay unchanged."""
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_sampling_ids(rows, images):
    if list(map(Path, images)) != [Path(row["image_path"]) for row in rows]:
        raise ValueError("Physical image order differs from canonical manifest")
    return [Path(row["image_id"] + ".jpg") for row in rows]


def derive_mixed_manifest(rows, protected, output):
    keys = {"image_ids": "image_id", "image_paths": "image_path",
            "identity_component_ids": "identity_component_id"}
    if any(row[key] in protected.get(kind, set()) for row in rows for kind, key in keys.items()):
        raise ValueError("Training manifest intersects protected identities")
    if any(row["role"] not in {"source_fit", "pad_adapt"} for row in rows):
        raise ValueError("Only source_fit and pad_adapt can train")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate training IDs")
    records = sorted(rows, key=lambda row: row["image_id"])
    with Path(output).open("x") as stream:
        stream.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    return records


def write_sampling_plan(rows, path):
    import numpy as np
    rows = sorted(rows, key=lambda row: row["image_id"])
    domains = np.asarray([int(row["role"] == "pad_adapt") for row in rows])
    if set(domains.tolist()) != {0, 1}:
        raise ValueError("Both source and clinical images are required")
    with Path(path).open("xb") as stream:
        np.savez(stream, image_ids=np.asarray([row["image_id"] for row in rows]),
                 domains=domains, probabilities=np.ones(len(rows), dtype=float))
    return digest(path)


def load_sampling_plan(path, expected, rows):
    import numpy as np
    if digest(path) != expected:
        raise ValueError("Sampling digest mismatch")
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files}
    ordered = sorted(rows, key=lambda row: row["image_id"])
    if result["image_ids"].tolist() != [row["image_id"] for row in ordered]:
        raise ValueError("Sampling identity order mismatch")
    result.update(source_quota=12, clinical_quota=4)
    return result


def validate_wrapper_config(config):
    required = {"base_config", "manifest", "manifest_sha256", "overlay_receipt",
                "overlay_receipt_sha256", "lr_multiplier", "training_gpu_hours_limit"}
    if required - config.keys():
        raise ValueError("Missing explicit wrapper configuration")
    config = {**config, "base_config": dict(config["base_config"])}
    if config.get("resume") and not config["base_config"].get("resume_checkpoint"):
        raise ValueError("Explicit resume checkpoint required")
    if not config.get("resume") and config["base_config"].get("resume_checkpoint"):
        raise ValueError("Resume checkpoint requires explicit resume")
    config["base_config"].setdefault("resume_checkpoint", "")
    if (config["base_config"].get("num_workers") != 0
            or config["lr_multiplier"] not in (1, 4)
            or not 0 < config["training_gpu_hours_limit"] <= 12):
        raise ValueError("Unsupported worker, LR multiplier or budget")
    return config


def validate_output_dir(path, resume=False):
    path = Path(path)
    if path.exists() and not resume:
        raise ValueError("Refusing output overwrite or implicit resume")
    if resume and not path.is_dir():
        raise ValueError("Resume directory missing")


class LocalScalarTracker:
    def __init__(self, path):
        self.path = Path(path)

    def start(self, params):
        with self.path.with_name("resolved-params.json").open("w") as stream:
            json.dump(params, stream, indent=2, allow_nan=False)

    def log_metric(self, key, value, step=None):
        if not math.isfinite(float(value)):
            raise ValueError("Scalar must be finite")
        with self.path.open("a") as stream:
            stream.write(json.dumps({"key": key, "value": value, "step": step}) + "\n")

    def log_artifact(self, path):
        pass

    def end(self):
        pass


def finite_gradient_norm(value):
    if not math.isfinite(float(value)):
        raise FloatingPointError("Nonfinite gradient; refusing optimizer step")
    return value


class CoverageState:
    def __init__(self, ids, domains, checkpoint_interval=500):
        self.ids, self.domains = list(ids), list(domains)
        self.counts = dict.fromkeys(self.ids, 0)
        self.steps = 0
        self.domain_counts = {"source": 0, "clinical": 0}

    def record_successful_batch(self, indices):
        for index in indices:
            self.counts[self.ids[index]] += 1
            self.domain_counts["clinical" if self.domains[index] else "source"] += 1
        self.steps += 1

    def checkpoint_payload(self):
        return {"counts": dict(self.counts), "steps": self.steps,
                "domain_counts": dict(self.domain_counts)}

    def restore(self, payload):
        if set(payload["counts"]) != set(self.ids):
            raise ValueError("Resume coverage identities differ")
        self.counts = dict(payload["counts"])
        self.steps = int(payload["steps"])
        self.domain_counts = dict(payload["domain_counts"])


if __name__ == "__main__":
    import argparse
    from overlay_ssl_runtime import run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    run(parser.parse_args().config)
