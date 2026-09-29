"""Every script that writes an evaluation/result JSON records its environment in a sidecar.

v7.1 recorded no versions or threads, and a T4 reproduction drift of ~1e-3 could
not be attributed. The registry below is the contract: a script is either WIRED
(it calls ``run_environment.write_sidecar`` for the JSON it writes) or EXEMPT
with a reason. A new JSON-writing script that is in neither fails the discovery
test. The block never goes INSIDE a result: guards pin results by the sha256 of
their bytes, and those bytes must not depend on the machine.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# script -> the JSON(s) that get a ``<name>.environment.json`` sidecar.
WIRED = {
    "scripts/v71_run.py": "units/*/result.json, holdout_grid.json, pretest.json",
    "scripts/v71_val.py": "selection.json, v71_final_eval*.json",
    "scripts/t4_sog_run.py": "t4_holdout_grid.json (via the shared block, was a local copy)",
    "scripts/t4_sog_val.py": "t4_selection.json, t4_final_eval*.json",
    "scripts/probe_convergence_check.py": "probe_convergence.json (logreg convergence diagnostic)",
    "scripts/c1_thread_repro_check.py": "c1_thread_repro.json (C1 BLAS-thread reproduction diagnostic)",
    "scripts/t4_repro_compare.py": "the comparison JSON (plus a_/b_environment read from the runs' sidecars)",
    "scripts/v71_extra_contrasts.py": "extra_contrasts.json",
    "scripts/c1_final_eval.py": "c1 final_eval.json",
    "scripts/c1_run_grid.py": "c1_bootstrap.json and c1_cells.json",
    "scripts/c1_cross_backbone.py": "c1_cross_backbone.json",
    "scripts/c1_select_variant.py": "the selection JSON",
    "scripts/c1_reevaluate_crop_gate.py": "crop_regate_report.json",
    "scripts/c1_verify_bank_reproduction.py": "the --report JSON",
    "scripts/c1_near_dup_embed.py": "the embedding near-duplicate JSON (floating point, GPU)",
    "scripts/pad_ufes_eval.py": "the PAD-UFES zero-shot evaluation JSON",
    "scripts/pad_ufes_addendum.py": "the PAD-UFES A71/s44 descriptive addendum JSON (docs/30 7)",
    "scripts/pad_ufes_fst_validation.py": "the PAD-UFES FST validation JSON",
    "scripts/pad_ufes_wiseft.py": "the PAD-UFES WiSE-FT alpha curve JSON (docs/30 8)",
    "scripts/validar_fitzpatrick17k.py": "the Fitzpatrick17k validation report",
    "scripts/diagnostico_fst_comparativa.py": "the FST diagnostic manifest",
    "scripts/t2_luminance_dice.py": "t2 luminance-Dice JSON",
    "scripts/t6_luminance_threshold.py": "t6 luminance-threshold JSON",
    "scripts/t3_c1_luminance.py": "t3 C1-v7 by luminance tercile JSON",
    "scripts/t7_r0_tercile_check.py": "t7 r0 teacher-check JSON",
    "scripts/t7_compare.py": "t7 comparison JSON",
    "scripts/seg_paired_bootstrap.py": "the paired segmentation bootstrap JSON",
    "plan-segmentacion/comun/evaluacion/evaluate_segmentation.py": "summary.json (via eval_report)",
    "scripts/cluster_frozen_bank.py": "bank.json",
    "scripts/kaggle_circle_stage2.py": "the CIRCLe Stage 2 result JSON",
    "scripts/kaggle_fst_manifest.py": "the FST manifest inferred on Kaggle",
    "scripts/release_parity.py": "parity.json",
    "scripts/run_segnorm_baseline.py": "provenance.json (its own 'environment' key is the older package fingerprint)",
    "scripts/s4_unet_train.py": "run.json of the finished run",
    "scripts/s_diag_predict.py": "run.json",
    "scripts/s4_dino_saliency.py": "run.json",
    "scripts/s4_predict_pool.py": "run.json",
    "scripts/c1_predict_masks.py": "the masks manifest",
    "plan-segmentacion/comun/evaluacion/fst_pseudo_manifest.py": "the FST manifest JSON (the CSV is untouched)",
    "plan-segmentacion/comun/evaluacion/compare_fst_manifests.py": "the manifest comparison JSON",
}

_GENERATED = "generates notebooks, tables or site data from committed files; must stay byte-identical"
_DATA_PREP = "data preparation (split, index, exclusion or manifest of inputs), no evaluated number"
# script -> why it writes JSON without the block.
EXEMPT = {
    "scripts/build_baseline_tests.py": _GENERATED,
    "scripts/build_finetune_tests.py": _GENERATED,
    "scripts/build_kaggle_diagnostico_fst.py": _GENERATED,
    "scripts/build_release_package.py": _GENERATED,
    "scripts/build_site_data.py": _GENERATED,
    "scripts/build_split_attestation.py": _GENERATED,
    "scripts/notebook_code_v5.py": _GENERATED,
    "plan-segmentacion/comun/evaluacion/build_results_table.py": _GENERATED,
    "scripts/c1_build_val_index.py": _DATA_PREP,
    "scripts/c1_finals_exclude.py": _DATA_PREP,
    "scripts/c1_selection_holdout.py": _DATA_PREP,
    "scripts/pad_ufes_manifest.py": _DATA_PREP,
    "scripts/kaggle_circle_bundle.py": _DATA_PREP,
    "scripts/c1_near_dup_check.py": "leak audit by integer pHash: no float, BLAS or thread dependence",
    "scripts/c1_pool_leak_check.py": "leak audit by lesion id: set arithmetic, nothing to drift",
    "scripts/add_ece_basic_ci.py": ("amends a committed single-evaluation JSON with arithmetic on its stored "
                                    "numbers; a block here would describe another machine than the one "
                                    "that produced them"),
    "scripts/v71_io.py": "the atomic writer helper; its callers stamp the block",
    "plan-segmentacion/comun/evaluacion/eval_report.py": "the writer helper; evaluate_segmentation passes the block",
    "scripts/run_stage2.py": "orchestrator state; each stage it launches writes its own result",
    "scripts/kaggle_run_bank.py": "orchestrator state of Kaggle sessions",
    "scripts/kaggle_push.py": "writes kernel metadata and notebooks for the Kaggle CLI",
    "scripts/check_release_guides.py": "fixture release.json for a guide smoke test",
    "scripts/finalize_release.py": "file sizes of the release, no evaluated number",
    "scripts/warm_hub_cache.py": "list of cached hub files",
}

SCANNED = [*sorted((ROOT / "scripts").glob("*.py")),
           *sorted((ROOT / "plan-segmentacion" / "comun" / "evaluacion").glob("*.py"))]
# A JSON file written: json.dump(, .write_text(json.dumps(...)) or a write_*json( helper.
WRITES_JSON = re.compile(r"json\.dump\(|write_text\(\s*json\.dumps|\bwrite_\w*json\(")



def _run_environment_calls(path: Path) -> set[str]:
    """Names of ``run_environment.<name>(...)`` calls in a script."""
    return {node.func.attr for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "run_environment"}


@pytest.mark.parametrize("script", sorted(WIRED))
def test_every_wired_script_writes_an_environment_sidecar(script):
    calls = _run_environment_calls(ROOT / script)
    assert "write_sidecar" in calls, f"{script} writes a result JSON without run_environment.write_sidecar()"
    assert "snapshot" not in calls, f"{script} puts the block inside a result, where it changes its sha256"


def test_every_json_writing_script_is_wired_or_exempt_with_a_reason():
    writers = {str(p.relative_to(ROOT)) for p in SCANNED if WRITES_JSON.search(p.read_text(encoding="utf-8"))}
    unclassified = sorted(writers - WIRED.keys() - EXEMPT.keys())
    assert not unclassified, f"write JSON but are neither WIRED nor EXEMPT: {unclassified}"


def test_the_registry_names_real_files_and_no_script_is_both():
    assert not WIRED.keys() & EXEMPT.keys()
    missing = [s for s in (*WIRED, *EXEMPT) if not (ROOT / s).is_file()]
    assert not missing, missing
    assert all(reason.strip() for reason in EXEMPT.values())


def test_the_detector_sees_a_multiline_write():
    assert WRITES_JSON.search('out.write_text(\n    json.dumps(doc), encoding="utf-8")')
    assert WRITES_JSON.search("eval_report.write_summary_json(path, m, meta)")
    assert not WRITES_JSON.search("print(json.dumps(doc))")


def _tracked_json() -> list[Path]:
    import subprocess

    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.json"], capture_output=True, text=True, check=True)
    return [ROOT / line for line in out.stdout.split()]


def test_no_committed_result_carries_the_block_inside():
    import json

    inside = []
    for path in _tracked_json():
        if path.name.endswith(".environment.json"):
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(doc, dict) and isinstance(doc.get("environment"), dict) and "schema" in doc["environment"]:
            inside.append(str(path.relative_to(ROOT)))
    assert not inside, f"environment block inside a hashed result: {inside}"


# Committed artifacts other guards or documents pin by sha256: the move to a
# sidecar must not change a single byte of them.
PINNED = {
    "resultados/clasificacion/v71/selection.json":
        "24e24e4ee6b97717e1bb79ac4e7dfa6447c69c7e40b222a5c40b02cb49af0ec1",
}


@pytest.mark.parametrize("rel", sorted(PINNED))
def test_pinned_committed_artifacts_keep_their_sha256(rel):
    import hashlib

    assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == PINNED[rel]


def _recorded_file_shas(node, found: list) -> None:
    """(path, sha256) pairs a committed JSON records for files: {path: sha} or {path:, sha256:}."""
    if isinstance(node, list):
        for item in node:
            _recorded_file_shas(item, found)
    if not isinstance(node, dict):
        return
    if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
        found.append((node["path"], node["sha256"]))
    for key, value in node.items():
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) and "/" in key:
            found.append((key, value))
        _recorded_file_shas(value, found)


def test_every_sha256_a_committed_json_records_for_a_committed_file_still_matches():
    import hashlib
    import json

    checked, stale = 0, []
    for path in _tracked_json():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        found: list = []
        _recorded_file_shas(doc, found)
        for rel, sha in found:
            target = ROOT / rel
            if not target.is_file():
                continue  # a pod path: not in the repository, nothing to recompute here
            checked += 1
            if hashlib.sha256(target.read_bytes()).hexdigest() != sha:
                stale.append(f"{path.relative_to(ROOT)} -> {rel}")
    assert checked >= 7 and not stale, stale
