"""Every script that writes an evaluation/result JSON embeds the environment block.

v7.1 recorded no versions or threads, and a T4 reproduction drift of ~1e-3 could
not be attributed. The registry below is the contract: a script is either WIRED
(it calls ``run_environment.snapshot`` for the JSON it writes) or EXEMPT with a
reason. A new JSON-writing script that is in neither fails the discovery test.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# script -> the JSON(s) whose top level carries ``environment``.
WIRED = {
    "scripts/v71_run.py": "units/*/result.json, holdout_grid.json, pretest.json",
    "scripts/v71_val.py": "selection.json, v71_final_eval*.json",
    "scripts/t4_sog_run.py": "t4_holdout_grid.json (via the shared block, was a local copy)",
    "scripts/t4_sog_val.py": "t4_selection.json, t4_final_eval*.json",
    "scripts/t4_repro_compare.py": "the comparison JSON (plus a_/b_environment of the runs)",
    "scripts/v71_extra_contrasts.py": "extra_contrasts.json",
    "scripts/c1_final_eval.py": "c1 final_eval.json",
    "scripts/c1_run_grid.py": "c1_bootstrap.json (c1_cells.json stays a pure cell map)",
    "scripts/c1_cross_backbone.py": "c1_cross_backbone.json",
    "scripts/c1_select_variant.py": "the selection JSON",
    "scripts/c1_reevaluate_crop_gate.py": "crop_regate_report.json",
    "scripts/c1_verify_bank_reproduction.py": "the --report JSON",
    "scripts/c1_near_dup_embed.py": "the embedding near-duplicate JSON (floating point, GPU)",
    "scripts/pad_ufes_eval.py": "the PAD-UFES zero-shot evaluation JSON",
    "scripts/pad_ufes_fst_validation.py": "the PAD-UFES FST validation JSON",
    "scripts/validar_fitzpatrick17k.py": "the Fitzpatrick17k validation report",
    "scripts/diagnostico_fst_comparativa.py": "the FST diagnostic manifest",
    "scripts/t2_luminance_dice.py": "t2 luminance-Dice JSON",
    "scripts/t6_luminance_threshold.py": "t6 luminance-threshold JSON",
    "scripts/t3_c1_luminance.py": "t3 C1-v7 by luminance tercile JSON",
    "scripts/seg_paired_bootstrap.py": "the paired segmentation bootstrap JSON",
    "plan-segmentacion/comun/evaluacion/evaluate_segmentation.py": "summary.json (via eval_report)",
    "scripts/cluster_frozen_bank.py": "bank.json (strata.json stays a map of backbones)",
    "scripts/kaggle_circle_stage2.py": "the CIRCLe Stage 2 result JSON",
    "scripts/kaggle_fst_manifest.py": "the FST manifest inferred on Kaggle",
    "scripts/release_parity.py": "parity.json",
    "scripts/run_segnorm_baseline.py": "provenance.json under run_environment ('environment' is its older fingerprint)",
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



def _calls_snapshot(path: Path) -> bool:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "snapshot" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "run_environment"):
            return True
    return False


@pytest.mark.parametrize("script", sorted(WIRED))
def test_every_wired_script_embeds_the_environment_block(script):
    assert _calls_snapshot(ROOT / script), f"{script} writes a result JSON without run_environment.snapshot()"


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
