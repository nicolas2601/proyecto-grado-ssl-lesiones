"""Prepare runs 024 and 025: run 023 replicated on seeds 43 and 44.

Run 023 (ResNet34 + PH2 copy-paste + IMA++ N=100 real, seed 42) is the only
positive result of the v8 segmentation line, and it rests on a single seed. The
copy-paste ablation already showed that a single-seed effect can reverse, so this
prepares the two replications needed to turn it into a three-seed claim.

Single factor changed with respect to 023: the random seed. Device differs by
necessity (one run per GPU) and is recorded as such, not as an experimental
factor.

This script only writes configs, gates and launchers. It launches nothing.
Intended to run on UNABIA, where the source config and gates live.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/home/jovyan/work/dermlip_slic_v8")
RUNTIME_ROOT = ROOT / "runtime_poolv2_20261022_001"
CFG_DIR = RUNTIME_ROOT / "plan-segmentacion/metodos-v8/launch-preparation/configs"
PRE_DIR = RUNTIME_ROOT / "preflight"

# Source of truth: the config and gates that run 023 actually used and completed with.
SRC_CFG = CFG_DIR / "dermlip.selection.runtime.20261028.023.json"
SRC_EXEC_GATE = PRE_DIR / "dermlip-selection-execution-gate-v2-20261028-023.json"
SRC_RUN_GATE = PRE_DIR / "dermlip-selection-runtime-gate-v2-20261028-023.json"

# Observed directly with nvidia-smi before preparing; both GPUs idle at 0 %, 2 MiB.
NVIDIA_SMI_OBSERVED_AT = "2026-10-31 16:42:49"

TRAIN_SCRIPT = ROOT / "scripts/train_dermlip_semi_023.py"
PYTHON_ENV = "/home/jovyan/work/proyectonp/.venv/bin/python"

REPLICAS = [
    {
        "run_num": "024",
        "seed": 43,
        "device": "cuda:0",
        "physical_gpu": 0,
        "run_name": (
            "dermlip-semi-supervised-v8-isic2018-2594-"
            "abl024-resnet34-ph2cp-imapp100-s43"
        ),
    },
    {
        "run_num": "025",
        "seed": 44,
        "device": "cuda:1",
        "physical_gpu": 1,
        "run_name": (
            "dermlip-semi-supervised-v8-isic2018-2594-"
            "abl025-resnet34-ph2cp-imapp100-s44"
        ),
    },
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha256(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare(replica: dict) -> dict:
    run_num = replica["run_num"]
    src_cfg = json.loads(SRC_CFG.read_text(encoding="utf-8"))
    src_exec = json.loads(SRC_EXEC_GATE.read_text(encoding="utf-8"))
    src_run = json.loads(SRC_RUN_GATE.read_text(encoding="utf-8"))

    out_root = f"{ROOT}/runs/{replica['run_name']}/selection/"
    if Path(out_root).exists():
        raise SystemExit(f"refusing to prepare {run_num}: {out_root} already exists")

    claim = (
        f"{out_root}claims/dermlip-selection-20261031-{run_num}.selection.claim.json"
    )
    exec_p = PRE_DIR / f"dermlip-selection-execution-gate-v2-20261031-{run_num}.json"
    run_p = PRE_DIR / f"dermlip-selection-runtime-gate-v2-20261031-{run_num}.json"
    cfg_rel = (
        "plan-segmentacion/metodos-v8/launch-preparation/configs/"
        f"dermlip.selection.runtime.20261031.{run_num}.json"
    )
    cfg_p = CFG_DIR / f"dermlip.selection.runtime.20261031.{run_num}.json"

    # The only experimental change: the seed. Everything else is carried over.
    profile = dict(src_cfg["profile"])
    profile["seed"] = replica["seed"]
    profile["device"] = replica["device"]
    profile["execution_claim_path"] = claim
    profile_sha = json_sha256(profile)

    exec_gate = dict(src_exec)
    exec_gate["gate_id"] = f"dermlip-selection-execution-20261031-{run_num}"
    exec_gate["profile"] = profile
    exec_gate["profile_sha256"] = profile_sha
    exec_gate["execution_claim_path"] = claim
    exec_gate["gpu_preflight"] = dict(src_exec["gpu_preflight"])
    exec_gate["gpu_preflight"]["selected_logical_device"] = replica["device"]
    exec_gate["gpu_preflight"]["selected_physical_gpu_index"] = replica["physical_gpu"]
    exec_gate["gpu_preflight"]["observed_at_nvidia_smi"] = NVIDIA_SMI_OBSERVED_AT
    exec_p.write_text(json.dumps(exec_gate, indent=2) + "\n", encoding="utf-8")
    exec_sha = sha256_file(exec_p)

    cfg = dict(src_cfg)
    cfg["preparation_id"] = f"dermlip-selection-runtime-20261031-{run_num}"
    cfg["artifact_path"] = cfg_rel
    cfg["supersedes"] = [src_cfg["artifact_path"]]
    paths = dict(src_cfg["paths"])
    paths["output_root"] = out_root
    paths["selection_gate"] = str(run_p)
    paths["execution_gate"] = str(exec_p)
    paths["execution_claim_path"] = claim
    paths["execution_gate_sha256"] = exec_sha
    cfg["paths"] = paths
    cfg["profile"] = profile
    # Carried over from 023 and still false: no run in this line is accredited for
    # patient independence. Recorded here so the new runs inherit the limitation
    # explicitly instead of silently.
    cfg["patient_independence_accredited"] = False
    cfg["training_started"] = False
    cfg_p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    cfg_sha = sha256_file(cfg_p)

    run_gate = dict(src_run)
    run_gate["gate_id"] = f"dermlip-selection-runtime-20261031-{run_num}"
    run_gate["runtime_config_relative_path"] = cfg_rel
    run_gate["runtime_config_sha256"] = cfg_sha
    run_gate["profile"] = profile
    run_gate["execution_claim_path"] = claim
    run_gate["gpu_preflight"] = dict(src_run["gpu_preflight"])
    run_gate["gpu_preflight"]["selected_logical_device"] = replica["device"]
    run_gate["gpu_preflight"]["selected_physical_gpu_index"] = replica["physical_gpu"]
    run_gate["gpu_preflight"]["observed_at_nvidia_smi"] = NVIDIA_SMI_OBSERVED_AT
    run_p.write_text(json.dumps(run_gate, indent=2) + "\n", encoding="utf-8")

    launcher = ROOT / f"launch_train_{run_num}_abl_resnet34_ph2cp_imapp100_s{replica['seed']}.sh"
    log = ROOT / f"runs/dermlip_abl{run_num}_console.log"
    launcher.write_text(
        "#!/bin/bash\n"
        "set -e\n"
        'export PYTHONPATH="/home/jovyan/work/proyectonp/repo_v8_canonical/src:'
        '/home/jovyan/work/dermlip_slic_v8:$PYTHONPATH"\n'
        "export CUDA_VISIBLE_DEVICES=0,1\n"
        f'CFG="{cfg_p}"\n'
        f'PYTHON="{PYTHON_ENV}"\n'
        f'SCRIPT="{TRAIN_SCRIPT}"\n'
        f'LOG="{log}"\n'
        "\n"
        "nohup $PYTHON $SCRIPT \\\n"
        '    --runtime-config "$CFG" \\\n'
        "    --backbone resnet34 \\\n"
        "    --flat-lr \\\n"
        "    --superpixel-affinity-weight 0.1 \\\n"
        "    --pseudo-label-threshold-high 0.65 \\\n"
        "    --pseudo-label-threshold-low 0.35 \\\n"
        "    --copy-paste-prob 0.5 \\\n"
        f"    --seed {replica['seed']} \\\n"
        f"    --device {replica['device']} \\\n"
        "    --execute \\\n"
        '    > "$LOG" 2>&1 &\n'
        "\n"
        f'echo "Launched {run_num} (seed {replica["seed"]}, {replica["device"]}) PID: $!"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)

    return {
        "run_num": run_num,
        "seed": replica["seed"],
        "device": replica["device"],
        "output_root": out_root,
        "config": str(cfg_p),
        "config_sha256": cfg_sha,
        "execution_gate": str(exec_p),
        "execution_gate_sha256": exec_sha,
        "runtime_gate": str(run_p),
        "profile_sha256": profile_sha,
        "launcher": str(launcher),
    }


def main() -> None:
    prepared = [prepare(replica) for replica in REPLICAS]
    print(json.dumps(prepared, indent=2))

    src_profile = json.loads(SRC_CFG.read_text(encoding="utf-8"))["profile"]
    print("\nDiff against 023 profile, per replica:")
    for item, replica in zip(prepared, REPLICAS):
        cfg = json.loads(Path(item["config"]).read_text(encoding="utf-8"))
        changed = {
            key: (src_profile.get(key), value)
            for key, value in cfg["profile"].items()
            if src_profile.get(key) != value
        }
        print(f"  {item['run_num']}: {sorted(changed)}")


if __name__ == "__main__":
    main()
