"""Prepare runs 026-031: the first single-factor ablation of the IMA++ cohort.

Neither existing comparison isolates IMA++. Runs 019-022 used
``train_dermlip_semi.py``, which gets copy-paste by importing
``dermssl.segmentation_methods.dermlip_slic.copy_paste``; runs 023-025 used
``train_dermlip_semi_023.py``, which carries its own 82-line
``apply_ph2_copy_paste`` (verified: that function is defined at
train_dermlip_semi_023.py:1172 and does not exist in train_dermlip_semi.py).
So 023-vs-020 conflates IMA++ with a re-implemented augmentation.

This lot varies exactly one factor, via ``--ima-batch`` on the 026 trainer:

    --ima-batch 0   16 PH2 + 0 IMA++ = 16 labelled   control
    --ima-batch 6   10 PH2 + 6 IMA++ = 16 labelled   IMA++ arm (reproduces 023)

The labelled batch is held at 16 in both arms on purpose. Dropping it to 10 for
the control would change labelled examples per step, which is a second factor.

Device assignment alternates across waves so that GPU is NOT perfectly
correlated with arm:

    wave 1  026 control cuda:0   027 IMA++ cuda:1
    wave 2  028 control cuda:1   029 IMA++ cuda:0
    wave 3  030 control cuda:0   031 IMA++ cuda:1

Both GPUs are the same model, so the risk is small, but assigning every control
to one GPU and every IMA++ arm to the other would make device and arm
indistinguishable by construction. Device is recorded as an operational fact,
never as an experimental factor.

This script writes configs, gates and launchers. It launches nothing, and it
never edits an existing gate: every gate is generated fresh from the config that
run 023 actually completed with.

Intended to run on UNABIA, where the source config, gates and pool manifests
live. The repository has no copy of the 023 config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/jovyan/work/dermlip_slic_v8")
RUNTIME_ROOT = ROOT / "runtime_poolv2_20261022_001"
CFG_DIR = RUNTIME_ROOT / "plan-segmentacion/metodos-v8/launch-preparation/configs"
PRE_DIR = RUNTIME_ROOT / "preflight"

# Source of truth: the config and gates run 023 actually used and completed with.
SRC_CFG = CFG_DIR / "dermlip.selection.runtime.20261028.023.json"
SRC_EXEC_GATE = PRE_DIR / "dermlip-selection-execution-gate-v2-20261028-023.json"
SRC_RUN_GATE = PRE_DIR / "dermlip-selection-runtime-gate-v2-20261028-023.json"

PYTHON_ENV = "/home/jovyan/work/proyectonp/.venv/bin/python"

# --- v3 source-disjoint pool pins -------------------------------------------
# Not hardcoded, on purpose. The pins already rotated once (-001 was discarded
# because it copied pool_contract verbatim and carried the v2 stage pins while
# its own count said 34115; the library validator does not check those fields, so
# the inconsistency would have reached a gate). Hardcoding them means every
# rotation needs a code edit, which is how a stale pin reaches a config.
#
# The pins live in a versioned JSON passed with --pins, and every value in it is
# re-checked against the files on disk before anything is written.
DEFAULT_PINS = Path(__file__).resolve().parents[1] / (
    "plan-segmentacion/metodos-v8/launch-preparation/pins/v3-pool-20261101-002.pins.json"
)


@dataclass(frozen=True)
class Pins:
    """Pinned inputs for one lot, loaded from a pins JSON."""

    pins_id: str
    pool_root: Path
    selection_manifest: Path
    selection_sha256: str
    selection_cohort_sha256: str
    selection_record_count: int
    selection_dataset_counts: dict
    pool_receipt: Path
    pool_receipt_sha256: str
    exclusion_receipt: Path
    exclusion_receipt_sha256: str
    cohort_attribution: Path
    attestation: Path
    attestation_sha256: str
    train_script: Path
    train_script_sha256: str

    @classmethod
    def load(cls, path: Path) -> "Pins":
        raw = json.loads(path.read_text(encoding="utf-8"))
        pool_root = Path(raw["pool_root"])
        sel, receipt = raw["selection_manifest"], raw["pool_receipt"]
        excl, attr = raw["exclusion_receipt"], raw["cohort_attribution"]
        att, trainer = raw["pool_use_attestation"], raw["train_script"]
        return cls(
            pins_id=raw["pins_id"],
            pool_root=pool_root,
            selection_manifest=pool_root / sel["name"],
            selection_sha256=sel["sha256"],
            selection_cohort_sha256=sel["cohort_sha256"],
            selection_record_count=int(sel["record_count"]),
            selection_dataset_counts=dict(sel["dataset_counts"]),
            pool_receipt=pool_root / receipt["name"],
            pool_receipt_sha256=receipt["sha256"],
            exclusion_receipt=pool_root / excl["name"],
            exclusion_receipt_sha256=excl["sha256"],
            cohort_attribution=pool_root / attr["name"],
            attestation=Path(att["path"]),
            attestation_sha256=att["sha256"],
            train_script=Path(trainer["path"]),
            train_script_sha256=trainer["sha256"],
        )

RUNS = [
    {"run_num": "026", "wave": 1, "seed": 42, "ima_batch": 0, "arm": "control", "device": "cuda:0", "gpu": 0},
    {"run_num": "027", "wave": 1, "seed": 42, "ima_batch": 6, "arm": "imapp", "device": "cuda:1", "gpu": 1},
    {"run_num": "028", "wave": 2, "seed": 43, "ima_batch": 0, "arm": "control", "device": "cuda:1", "gpu": 1},
    {"run_num": "029", "wave": 2, "seed": 43, "ima_batch": 6, "arm": "imapp", "device": "cuda:0", "gpu": 0},
    {"run_num": "030", "wave": 3, "seed": 44, "ima_batch": 0, "arm": "control", "device": "cuda:0", "gpu": 0},
    {"run_num": "031", "wave": 3, "seed": 44, "ima_batch": 6, "arm": "imapp", "device": "cuda:1", "gpu": 1},
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha256(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def observe_gpus() -> tuple[str, str]:
    """Record the GPU state at preparation time instead of hardcoding it.

    A gate that carries a stale GPU observation asserts something that was true
    for a different moment. Reading it here keeps the claim honest.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), out


def preflight(pins: Pins) -> None:
    """Abort before writing anything if any pinned input fails to match."""
    problems: list[str] = []

    for path in (SRC_CFG, SRC_EXEC_GATE, SRC_RUN_GATE, pins.train_script,
                 pins.selection_manifest, pins.pool_receipt, pins.exclusion_receipt,
                 pins.cohort_attribution, pins.attestation):
        if not path.exists():
            problems.append(f"missing input: {path}")
    if problems:
        raise SystemExit("preflight failed:\n  " + "\n  ".join(problems))

    checks = [
        ("v3 selection manifest", pins.selection_manifest, pins.selection_sha256),
        ("v3 pool receipt", pins.pool_receipt, pins.pool_receipt_sha256),
        ("v3 exclusion receipt", pins.exclusion_receipt, pins.exclusion_receipt_sha256),
        ("v3 pool-use attestation", pins.attestation, pins.attestation_sha256),
        ("026 trainer", pins.train_script, pins.train_script_sha256),
    ]
    for label, path, expected in checks:
        actual = sha256_file(path)
        if actual != expected:
            problems.append(f"{label} sha256 mismatch\n      expected {expected}\n      actual   {actual}")

    # The library validator requires the manifest to live under the pool root and
    # to be named exactly selection.manifest.json.
    if pins.selection_manifest.name != "selection.manifest.json":
        problems.append("v3 manifest must be named selection.manifest.json")
    if pins.selection_manifest.parent != pins.pool_root:
        problems.append(f"v3 manifest must live directly under {pins.pool_root}")

    manifest = json.loads(pins.selection_manifest.read_text(encoding="utf-8"))
    if manifest.get("count") != pins.selection_record_count:
        problems.append(
            f"v3 manifest count {manifest.get('count')} != pinned {pins.selection_record_count}"
        )
    records = manifest.get("records") or []
    if len(records) != pins.selection_record_count:
        problems.append("v3 manifest len(records) does not match its own count")
    if manifest.get("source_cohort_disjoint_from_evaluation") is not True:
        problems.append("v3 manifest does not declare source_cohort_disjoint_from_evaluation")

    counts: dict[str, int] = {}
    for record in records:
        key = record.get("dataset")
        counts[key] = counts.get(key, 0) + 1
    if counts != pins.selection_dataset_counts:
        problems.append(
            f"v3 manifest dataset_counts {counts} != pinned {pins.selection_dataset_counts}"
        )

    # The -001 build passed the library validator while carrying the v2 stage pins
    # inside pool_contract, because the validator does not look at those fields.
    # Check them here so a desynchronised contract cannot reach a gate.
    contract = manifest.get("pool_contract") or {}
    if "stage_count" in contract and contract["stage_count"] != manifest.get("count"):
        problems.append(
            f"pool_contract.stage_count {contract['stage_count']} disagrees with "
            f"manifest count {manifest.get('count')}"
        )
    if ("stage_cohort_sha256" in contract
            and contract["stage_cohort_sha256"] != pins.selection_cohort_sha256):
        problems.append("pool_contract.stage_cohort_sha256 disagrees with the pinned cohort digest")
    if "stage_dataset_counts" in contract and contract["stage_dataset_counts"] != counts:
        problems.append("pool_contract.stage_dataset_counts disagrees with the manifest records")

    # Source-cohort disjointness is only auditable if the per-image attribution is
    # persisted and complete. Anything left unattributed means the claim rests on
    # an aggregate nobody can re-derive.
    attribution = json.loads(pins.cohort_attribution.read_text(encoding="utf-8"))
    if attribution.get("evaluation_unattributed"):
        problems.append(
            f"{len(attribution['evaluation_unattributed'])} evaluation images have no "
            "source-cohort attribution; disjointness is not auditable"
        )
    attributed = attribution.get("evaluation_attributed")
    split_size = attribution.get("evaluation_split_size")
    if attributed != split_size:
        problems.append(
            f"attribution covers {attributed} of {split_size} evaluation images"
        )

    # The exclusion accredits source-cohort disjointness, NOT patient
    # independence. The library rejects an attestation that claims otherwise, and
    # so do we: these two must stay false.
    receipt = json.loads(pins.pool_receipt.read_text(encoding="utf-8"))
    if receipt.get("patient_independence_accredited") is not False:
        problems.append("v3 receipt must keep patient_independence_accredited false")

    for run in RUNS:
        out_root = ROOT / f"runs/{run_name(run)}/selection"
        if out_root.exists():
            problems.append(f"output already exists for {run['run_num']}: {out_root}")

    if problems:
        raise SystemExit("preflight failed:\n  " + "\n  ".join(problems))


def run_name(run: dict) -> str:
    return (
        "dermlip-semi-supervised-v8-isic2018-2594-"
        f"abl{run['run_num']}-resnet34-ph2cp-ima{run['ima_batch']}-s{run['seed']}"
    )


def prepare(run: dict, observed_at: str, pins: Pins) -> dict:
    run_num = run["run_num"]
    src_cfg = json.loads(SRC_CFG.read_text(encoding="utf-8"))
    src_exec = json.loads(SRC_EXEC_GATE.read_text(encoding="utf-8"))
    src_run = json.loads(SRC_RUN_GATE.read_text(encoding="utf-8"))

    out_root = f"{ROOT}/runs/{run_name(run)}/selection/"
    claim = f"{out_root}claims/dermlip-selection-20261101-{run_num}.selection.claim.json"
    exec_p = PRE_DIR / f"dermlip-selection-execution-gate-v3-20261101-{run_num}.json"
    run_p = PRE_DIR / f"dermlip-selection-runtime-gate-v3-20261101-{run_num}.json"
    cfg_rel = (
        "plan-segmentacion/metodos-v8/launch-preparation/configs/"
        f"dermlip.selection.runtime.20261101.{run_num}.json"
    )
    cfg_p = CFG_DIR / f"dermlip.selection.runtime.20261101.{run_num}.json"

    profile = dict(src_cfg["profile"])
    profile["seed"] = run["seed"]
    profile["device"] = run["device"]
    profile["execution_claim_path"] = claim
    # Recorded explicitly so the receipts carry the experimental factor. This is
    # exactly what 023 lacked: no receipt of 023 accredited its IMA++ cohort.
    profile["ima_batch"] = run["ima_batch"]
    profile["ph2_batch"] = 16 - run["ima_batch"]
    profile["labelled_batch_total"] = 16
    profile["arm"] = run["arm"]
    profile_sha = json_sha256(profile)

    exec_gate = dict(src_exec)
    exec_gate["gate_id"] = f"dermlip-selection-execution-20261101-{run_num}"
    exec_gate["profile"] = profile
    exec_gate["profile_sha256"] = profile_sha
    exec_gate["execution_claim_path"] = claim
    exec_gate["train_script"] = str(pins.train_script)
    exec_gate["train_script_sha256"] = pins.train_script_sha256
    exec_gate["gpu_preflight"] = dict(src_exec["gpu_preflight"])
    exec_gate["gpu_preflight"]["selected_logical_device"] = run["device"]
    exec_gate["gpu_preflight"]["selected_physical_gpu_index"] = run["gpu"]
    exec_gate["gpu_preflight"]["observed_at_nvidia_smi"] = observed_at
    exec_p.write_text(json.dumps(exec_gate, indent=2) + "\n", encoding="utf-8")
    exec_sha = sha256_file(exec_p)

    cfg = dict(src_cfg)
    cfg["preparation_id"] = f"dermlip-selection-runtime-20261101-{run_num}"
    cfg["artifact_path"] = cfg_rel
    cfg["supersedes"] = [src_cfg["artifact_path"]]

    paths = dict(src_cfg["paths"])
    paths["pool_manifest"] = str(pins.selection_manifest)
    paths["pool_use_attestation"] = str(pins.attestation)
    paths["output_root"] = out_root
    paths["selection_gate"] = str(run_p)
    paths["execution_gate"] = str(exec_p)
    paths["execution_claim_path"] = claim
    paths["execution_gate_sha256"] = exec_sha
    cfg["paths"] = paths

    bindings = dict(src_cfg["bindings"])
    bindings["pool_manifest"] = {
        "path": str(pins.selection_manifest),
        "sha256": pins.selection_sha256,
        "cohort_sha256": pins.selection_cohort_sha256,
        "record_count": pins.selection_record_count,
        "dataset_counts": dict(pins.selection_dataset_counts),
        "stage": "selection",
    }
    bindings["pool_use_attestation"] = {
        "path": str(pins.attestation),
        "sha256": pins.attestation_sha256,
    }
    cfg["bindings"] = bindings

    cfg["profile"] = profile
    cfg["train_script"] = str(pins.train_script)
    cfg["train_script_sha256"] = pins.train_script_sha256
    # Source-cohort disjointness is its own claim and does not upgrade either of
    # the two below. Carried over as false, explicitly, so the new runs inherit
    # the limitation instead of silently dropping it.
    cfg["source_cohort_disjoint_from_evaluation"] = True
    cfg["patient_independence_accredited"] = False
    cfg["holdout_novelty"] = False
    cfg["training_started"] = False
    cfg_p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    cfg_sha = sha256_file(cfg_p)

    run_gate = dict(src_run)
    run_gate["gate_id"] = f"dermlip-selection-runtime-20261101-{run_num}"
    run_gate["runtime_config_relative_path"] = cfg_rel
    run_gate["runtime_config_sha256"] = cfg_sha
    run_gate["profile"] = profile
    run_gate["execution_claim_path"] = claim
    run_gate["pool_manifest_sha256"] = pins.selection_sha256
    run_gate["pool_use_attestation_sha256"] = pins.attestation_sha256
    run_gate["gpu_preflight"] = dict(src_run["gpu_preflight"])
    run_gate["gpu_preflight"]["selected_logical_device"] = run["device"]
    run_gate["gpu_preflight"]["selected_physical_gpu_index"] = run["gpu"]
    run_gate["gpu_preflight"]["observed_at_nvidia_smi"] = observed_at
    run_p.write_text(json.dumps(run_gate, indent=2) + "\n", encoding="utf-8")

    log = ROOT / f"runs/dermlip_abl{run_num}_console.log"
    launcher = ROOT / f"launch_train_{run_num}_ima{run['ima_batch']}_s{run['seed']}.sh"
    launcher.write_text(
        "#!/bin/bash\n"
        "set -e\n"
        'export PYTHONPATH="/home/jovyan/work/proyectonp/repo_v8_canonical/src:'
        '/home/jovyan/work/dermlip_slic_v8:$PYTHONPATH"\n'
        "export CUDA_VISIBLE_DEVICES=0,1\n"
        f'CFG="{cfg_p}"\n'
        f'PYTHON="{PYTHON_ENV}"\n'
        f'SCRIPT="{pins.train_script}"\n'
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
        f"    --ima-batch {run['ima_batch']} \\\n"
        f"    --seed {run['seed']} \\\n"
        f"    --device {run['device']} \\\n"
        "    --execute \\\n"
        '    > "$LOG" 2>&1 &\n'
        "\n"
        f'echo "Launched {run_num} ({run["arm"]}, ima_batch={run["ima_batch"]}, '
        f'seed {run["seed"]}, {run["device"]}) PID: $!"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)

    return {
        "run_num": run_num,
        "wave": run["wave"],
        "arm": run["arm"],
        "seed": run["seed"],
        "ima_batch": run["ima_batch"],
        "device": run["device"],
        "run_name": run_name(run),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pins",
        type=Path,
        default=DEFAULT_PINS,
        help=(
            "Pins JSON for this lot. Defaults to the repository copy "
            f"({DEFAULT_PINS.name}). Pass a different file when the pool rotates "
            "instead of editing this script."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the preflight and report, without writing any config or gate",
    )
    args = parser.parse_args()

    if not args.pins.is_file():
        raise SystemExit(f"pins file not found: {args.pins}")
    pins = Pins.load(args.pins)
    print(f"pins: {pins.pins_id}")
    print(f"  from {args.pins}")
    print(f"  pool root {pins.pool_root}")
    print(f"  trainer   {pins.train_script}")
    print()

    preflight(pins)
    observed_at, gpu_state = observe_gpus()
    print(f"nvidia-smi at {observed_at} UTC:")
    for line in gpu_state.splitlines():
        print(f"  {line}")
    print()

    if args.dry_run:
        print("preflight OK. --dry-run set: nothing written.")
        return

    prepared = [prepare(run, observed_at, pins) for run in RUNS]
    print(json.dumps(prepared, indent=2))

    src_profile = json.loads(SRC_CFG.read_text(encoding="utf-8"))["profile"]
    print("\nProfile diff against 023, per run:")
    for item in prepared:
        cfg = json.loads(Path(item["config"]).read_text(encoding="utf-8"))
        changed = sorted(
            key for key, value in cfg["profile"].items() if src_profile.get(key) != value
        )
        print(f"  {item['run_num']} ({item['arm']}): {changed}")

    print("\nConfig sha256, for the launch record:")
    for item in prepared:
        print(f"  {item['run_num']}  ima_batch={item['ima_batch']}  seed={item['seed']}  "
              f"{item['device']}  {item['config_sha256']}")


if __name__ == "__main__":
    main()
