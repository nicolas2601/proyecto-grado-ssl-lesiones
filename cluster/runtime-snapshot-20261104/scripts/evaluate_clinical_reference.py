"""Patient-excluded native PAD development readout, never independent validation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATHS = json.loads((ROOT / "config/paths.json").read_text())
FROZEN = Path(PATHS["frozen_experiment_root"])
sys.path[:0] = [str(ROOT / "scripts"), str(FROZEN / "scripts"), str(FROZEN / "code/src")]
from native_reference import NATIVE_CLASSES, TEMPERATURE, feature_views, patient_excluded_vote
from evaluate_final_refit import load_encoder
from final_readout_helpers import pad_diagnoses
from phone_contract import digest, verify_source_snapshot
from dermssl.pretrain.image_input import load_rgb
from dermssl.phone_readout import neighbors, classification_summary
from dermssl.metrics import ece

MODELS = ("S0", "refit")
VIEWS = ("cls", "mean_patch", "unit_concat")
EXPECTED_FILES = {f"{model}_{view}.npz" for model in MODELS for view in VIEWS}


def load_inputs():
    protocol = json.loads((ROOT / "config/clinical-development-protocol-v1.json").read_text())
    manifest = ROOT / "inputs/pad-adapt-image-only.jsonl"
    if digest(manifest) != protocol["cohort"]["manifest_sha256"]:
        raise ValueError("Canonical PAD adaptation identities changed")
    rows = sorted((json.loads(line) for line in manifest.read_text().splitlines()), key=lambda row: row["image_id"])
    if (len(rows) != 1141 or len({row["image_id"] for row in rows}) != 1141
            or len({row["identity_component_id"] for row in rows}) != 686):
        raise ValueError("Authorized clinical cohort identities or patient counts changed")
    audit = json.loads((ROOT / "evidence/pad-adapt-content-audit.json").read_text())
    if any(item["cross_patient_groups"] for item in audit["aliases"].values()):
        raise ValueError("Cross-patient content aliases require a declared exclusion rule")
    fingerprints = json.loads((ROOT / "evidence/pad-adapt-content-fingerprints.json").read_text())
    expected = {row["image_id"]: row["byte_sha256"] for row in fingerprints}
    if set(expected) != {row["image_id"] for row in rows}:
        raise ValueError("Image-content audit does not cover the clinical cohort")
    for row in rows:
        if digest(row["image_path"]) != expected[row["image_id"]]:
            raise ValueError("An authorized clinical image changed after content audit")
    verify_source_snapshot()
    return protocol, rows, expected


def extract_blocks(model, rows):
    import torch
    from torchvision import transforms as T
    transform = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                           T.CenterCrop(224), T.ToTensor(),
                           T.Normalize((.485, .456, .406), (.229, .224, .225))])
    cls, patch = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            batch = torch.stack([transform(load_rgb(row["image_path"])) for row in rows[start:start + 32]]).to("cuda:0")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.forward_features(batch)
                current = output["x_norm_clstoken"].float()
                mean_patch = output["x_norm_patchtokens"].float().mean(dim=1)
                if start == 0:
                    torch.testing.assert_close(current, model(batch).float(), rtol=0, atol=0)
            cls.append(current.cpu().numpy())
            patch.append(mean_patch.cpu().numpy())
    return feature_views(np.concatenate(cls), np.concatenate(patch))


def export_neighbors(protocol, rows, expected, output):
    ids = np.asarray([row["image_id"] for row in rows])
    groups = np.asarray([row["identity_component_id"] for row in rows])
    files = {}
    for name in MODELS:
        weights = Path(PATHS["models"][name])
        if digest(weights) != protocol["encoders"][name]:
            raise ValueError("Frozen encoder checkpoint changed")
        model = load_encoder(weights)
        features = extract_blocks(model, rows)
        for view in VIEWS:
            indices, similarities = neighbors(features[view], features[view], ids, ids, 20, groups, groups)
            destination = output / f"{name}_{view}.npz"
            np.savez_compressed(destination, features=features[view], image_ids=ids, groups=groups,
                                indices=indices, similarities=similarities)
            files[destination.name] = digest(destination)
            print(json.dumps({"event": "blind_neighbors_frozen", "encoder": name, "feature_view": view,
                              "queries": len(rows), "dimensions": features[view].shape[1]}), flush=True)
        del model
    for row in rows:
        if digest(row["image_path"]) != expected[row["image_id"]]:
            raise ValueError("Clinical image bytes changed during feature extraction")
    seal = {"files": files, "encoder_sha256": protocol["encoders"],
            "protocol_sha256": digest(ROOT / "config/clinical-development-protocol-v1.json"),
            "manifest_sha256": protocol["cohort"]["manifest_sha256"],
            "diagnoses_used_for_neighbor_selection": False, "canonical_roles_regenerated": False}
    (output / "feature-neighbor-seal.json").write_text(json.dumps(seal, indent=2))


def read_development_labels(output, rows, protocol, paths):
    path = output / "feature-neighbor-seal.json"
    if not path.is_file():
        raise ValueError("Clinical diagnoses require the frozen feature-neighbor seal")
    seal = json.loads(path.read_text())
    if set(seal["files"]) != EXPECTED_FILES:
        raise ValueError("Clinical encoder/feature comparison must be complete before scoring")
    for name, expected in seal["files"].items():
        if digest(output / name) != expected:
            raise ValueError("Frozen clinical feature/neighbor cache changed")
    if (seal["protocol_sha256"] != digest(ROOT / "config/clinical-development-protocol-v1.json")
            or digest(paths["metadata_csv"]) != protocol["cohort"]["metadata_sha256"]):
        raise ValueError("Clinical protocol or diagnostic metadata changed")
    native, _ = pad_diagnoses(rows, paths["metadata_csv"])
    if set(native) != set(NATIVE_CLASSES):
        raise ValueError("Development bank must represent all six native diagnoses")
    return np.asarray(native)


def score_models(protocol, rows, output):
    labels = read_development_labels(output, rows, protocol, PATHS)
    ids = np.asarray([row["image_id"] for row in rows])
    groups = np.asarray([row["identity_component_id"] for row in rows])
    result = {"scope": "Transductive clinical-reference DEVELOPMENT, not independent or clinical validation",
              "native_classes": list(NATIVE_CLASSES), "images": len(rows), "patient_groups": len(set(groups)),
              "all_images_encoded_and_queried": True, "encoder_training": False, "system_wholly_label_free": False,
              "query_patient_excluded_from_neighbors_and_priors": True,
              "canonical_roles_regenerated": False, "old_tests_used_for_selection": False,
              "representation_exposure": protocol["representation_exposure"],
              "image_class_counts": {label: int((labels == label).sum()) for label in NATIVE_CLASSES},
              "patient_class_counts": {label: len(set(groups[labels == label])) for label in NATIVE_CLASSES},
              "results": {}}
    truth = np.asarray([NATIVE_CLASSES.index(label) for label in labels])
    for name in MODELS:
        result["results"][name] = {}
        for view in VIEWS:
            with np.load(output / f"{name}_{view}.npz", allow_pickle=False) as saved:
                cache = {key: saved[key].copy() for key in saved.files}
            if not np.array_equal(cache["image_ids"], ids) or not np.array_equal(cache["groups"], groups):
                raise ValueError("Clinical cache identities differ from the canonical cohort")
            scores, metrics = {}, {}
            for rule, balanced in (("balanced", True), ("plain", False)):
                scores[rule] = patient_excluded_vote(cache["indices"], cache["similarities"], labels, groups, groups, balanced)
                summary = classification_summary(labels, scores[rule], NATIVE_CLASSES)
                summary["ece"] = float(ece(truth, scores[rule], 15))
                metrics[rule] = summary
            np.savez_compressed(output / f"{name}_{view}_scores.npz", **scores, truth=labels, image_ids=ids, groups=groups)
            result["results"][name][view] = metrics
    return result


def track(result, output):
    import wandb
    destination = json.loads((FROZEN / "evidence/wandb-project.json").read_text())
    if destination["access"] != "PRIVATE":
        raise ValueError("Clinical development metrics require private tracking")
    scalars = {f"{model}/{view}/{rule}/{metric}": value
               for model, views in result["results"].items() for view, rules in views.items()
               for rule, metrics in rules.items() for metric, value in metrics.items() if isinstance(value, (float, int))}
    settings = wandb.Settings(disable_code=True, disable_git=True, save_code=False, console="off",
                              x_disable_meta=True, x_disable_stats=True, x_disable_machine_info=True)
    with wandb.init(entity=destination["entity"], project=destination["project"], mode="online",
                    name="native-six-clinical-reference-development", group="clf1-clinical-reference-20261024",
                    config={"development_only": True, "clinical_validation": False, "encoder_training": False,
                            "diagnostic_reference": True, "query_images": result["images"],
                            "seal_sha256": digest(output / "feature-neighbor-seal.json")}, settings=settings) as run:
        run.log(scalars)
        result["wandb_url"] = run.url


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    protocol, rows, expected = load_inputs()
    output = ROOT / "results/native-six-development-v1"
    if (output / "development-results.json").exists():
        raise FileExistsError("Preserve the existing development report; register a new version for changes")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if not args.score_only:
        output.mkdir(exist_ok=False)
        export_neighbors(protocol, rows, expected, output)
    result = score_models(protocol, rows, output)
    track(result, output)
    (output / "development-results.json").write_text(json.dumps(json_value(result), indent=2, allow_nan=False))
    print(json.dumps(json_value(result), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
