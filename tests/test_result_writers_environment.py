"""Every script that writes an evaluation/result JSON embeds the environment block.

v7.1 recorded no versions or threads, and a T4 reproduction drift of ~1e-3 could
not be attributed. The registry below is the contract: a script is either WIRED
(it calls ``run_environment.snapshot`` for the JSON it writes) or EXEMPT with a
reason. A new JSON-writing script that is in neither fails the discovery test.
"""

import ast
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
    "scripts/seg_paired_bootstrap.py": "the paired segmentation bootstrap JSON",
    "plan-segmentacion/comun/evaluacion/evaluate_segmentation.py": "summary.json (via eval_report)",
}


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
