"""Build configs, label groups and the two GPU queues for the 2026-10-29 follow-up batch.

Run once on the cluster: python scripts/build_followup_queue_20261029.py [--dry-run]
Refuses to overwrite anything that already exists.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/arms"))
from class_balance_20261029 import balanced_probabilities  # noqa: E402

STATE = ROOT / "results/ssl-followup-20261029"
CONFIGS = ROOT / "config/followup-20261029"
OVERLAY = ROOT / "config/ssl-overlay-v1"
MANIFEST = OVERLAY / "mixed-fit.jsonl"
DATA = Path("/home/jovyan/work")
PURE, GUIDED = "SSL puro", "SSL con muestreo guiado por etiquetas"
STEPS = {"smoke": 40, "replay": 40, "full": 6000}
CONFIG_FILE = {"smoke": "smoke.json", "replay": "replay.json", "full": "run.json"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def out_dirs(arm_id):
    return {kind: f"runs/{arm_id}-{kind}-20261029" for kind in STEPS}


def write_new(path, text):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_json(path, value):
    write_new(path, json.dumps(value, indent=2) + "\n")


def build_groups(rows):
    """Label groups used ONLY to choose images: source = malignant flag, clinical = PAD diagnosis."""
    target = {}
    for year in ("isic2019", "isic2020"):
        for record in csv.DictReader(open(DATA / f"datos/{year}/train-metadata.csv")):
            target[record["isic_id"]] = "mal" if record["target"] == "1" else "ben"
    diagnosis = {record["img_id"].removesuffix(".png"): record["diagnostic"]
                 for record in csv.DictReader(open(DATA / "pad_ufes_20/metadata.csv"))}
    groups = {}
    for row in rows:
        if row["role"] == "source_fit":
            groups[row["image_id"]] = target.get(row["image_id"], "ben")
        else:
            groups[row["image_id"]] = diagnosis[row["image_id"]]
    return groups


def write_balanced_inputs(rows):
    groups = build_groups(rows)
    groups_path = CONFIGS / "balance-groups.json"
    write_json(groups_path, groups)
    ordered = sorted(rows, key=lambda row: row["image_id"])
    plan = CONFIGS / "mixed-balanced.npz"
    with plan.open("xb") as stream:
        np.savez(stream, image_ids=np.asarray([row["image_id"] for row in ordered]),
                 domains=np.asarray([int(row["role"] == "pad_adapt") for row in ordered]),
                 probabilities=np.asarray(balanced_probabilities(ordered, groups), dtype=float))
    return groups_path, plan


def dino_configs(arm_id, seed, balance=None):
    configs = {}
    for kind in STEPS:
        config = json.loads((ROOT / "config/dino-ibot-v1" / CONFIG_FILE[kind]).read_text())
        config["seed"] = config["base_config"]["seed"] = seed
        config["base_config"]["output_dir"] = str(ROOT / out_dirs(arm_id)[kind])
        if balance:
            config["balance_groups"], config["balance_groups_sha256"] = str(balance), digest(balance)
        configs[kind] = config
    return configs


def overlay_configs(arm_id, plan):
    configs = {}
    for kind, source in (("smoke", "smoke.json"), ("replay", "smoke.json"), ("full", "A-s42.json")):
        config = json.loads((OVERLAY / source).read_text())
        config["base_config"]["output_dir"] = str(ROOT / out_dirs(arm_id)[kind])
        config["base_config"]["sampling_plan"] = str(plan)
        config["base_config"]["sampling_plan_sha256"] = digest(plan)
        configs[kind] = config
    return configs


def pretext_configs(arm_id, seed):
    base = json.loads((ROOT / "config/dino-ibot-v1/run.json").read_text())
    configs = {}
    for kind, steps in STEPS.items():
        configs[kind] = {"manifest": base["manifest"], "manifest_sha256": base["manifest_sha256"],
                         "manifest_roles": ["source_fit", "pad_adapt"], "batch_size": 16,
                         "base_config": {"initial_weights": base["base_config"]["initial_weights"],
                                         "output_dir": str(ROOT / out_dirs(arm_id)[kind]),
                                         "max_steps": steps, "seed": seed}}
    return configs


def arm_entry(arm_id, seed, script, label, section, kind_text, sources, guided=False, note=""):
    return {"id": arm_id, "tool": "projector", "script": script, "config_dir": f"config/followup-20261029/{arm_id}",
            "seed": seed, "label": label, "section": section, "type": kind_text,
            "encoder_uses_diagnostic_labels": guided, "label_guided_sampling": guided,
            "source_files": sources, "status_text": note or "exploratorio (runner de seguimiento)",
            "smoke_out": out_dirs(arm_id)["smoke"], "replay_out": out_dirs(arm_id)["replay"],
            "full_out": out_dirs(arm_id)["full"], "status": "pending", "error": None}


def plan_arms(rows):
    groups_path, plan = write_balanced_inputs(rows)
    dino = "scripts/arms/dino_ibot_followup_20261029.py"
    dino_src = ["scripts/dino_ibot_runtime.py", "scripts/arms/class_balance_20261029.py"]
    pretext = lambda name: [f"scripts/arms/{name}_20261029_runtime.py"]
    arms = []
    for seed in (43, 44):
        arm_id = f"dino-ibot-s{seed}"
        arms.append((0, arm_entry(arm_id, seed, dino, f"DINO + iBOT (EMA), semilla {seed}", PURE,
                                  "SSL puro (exploratorio)", dino_src), dino_configs(arm_id, seed)))
    note = "SSL con muestreo guiado por etiquetas: las etiquetas eligen imágenes, no entran a la pérdida; PAD usa las mismas etiquetas de evaluación"
    arms.append((0, arm_entry("dino-ibot-cb-s42", 42, dino, "DINO + iBOT + muestreo balanceado por clase", GUIDED,
                              "SSL guiado por etiquetas (exploratorio)", dino_src, True, note),
                 dino_configs("dino-ibot-cb-s42", 42, groups_path)))
    arms.append((1, arm_entry("simsiam-cb-s42", 42, "scripts/run_overlay_ssl.py", "SimSiam + muestreo balanceado por clase",
                              GUIDED, "SSL guiado por etiquetas (exploratorio)",
                              ["scripts/run_overlay_ssl.py", "scripts/overlay_ssl_runtime.py"], True, note),
                 overlay_configs("simsiam-cb-s42", plan)))
    specs = [(1, "masked-pixel", "Reconstrucción de píxeles enmascarados (estilo MAE)"),
             (1, "rotation", "Predicción de rotación (estilo pretexto)"),
             (0, "moco-v3", "Contraste con momentum (estilo MoCo-v3)")]
    for gpu, name, label in specs:
        arm_id = f"{name}-s42"
        helper = ROOT / f"scripts/arms/{name.replace('-', '_')}_20261029_helpers.py"
        helpers = [str(helper.relative_to(ROOT))] if helper.is_file() else []
        runtime = f"scripts/arms/{name.replace('-', '_')}_20261029_runtime.py"
        arms.append((gpu, arm_entry(arm_id, 42, runtime, label, PURE, "SSL puro (exploratorio)", helpers), pretext_configs(arm_id, 42)))
    return arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    if args.dry_run:
        groups = build_groups(rows)
        print(json.dumps({"rows": len(rows), "groups": {g: list(groups.values()).count(g) for g in sorted(set(groups.values()))}}))
        return
    queues = {0: [], 1: []}
    for gpu, entry, configs in plan_arms(rows):
        for kind, config in configs.items():
            write_json(ROOT / entry["config_dir"] / CONFIG_FILE[kind], config)
        queues[gpu].append(entry)
    for gpu, entries in queues.items():
        queue = STATE / f"gpu{gpu}/queue.json"
        if json.loads(queue.read_text()):
            raise RuntimeError(f"queue gpu{gpu} is not empty; refusing to overwrite")
        queue.write_text(json.dumps(copy.deepcopy(entries), indent=2, ensure_ascii=False) + "\n")
        print(gpu, [entry["id"] for entry in entries])


if __name__ == "__main__":
    main()
