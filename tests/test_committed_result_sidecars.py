"""Committed results never lose their environment, and no sidecar outlives its result.

Every committed JSON produced by a script of the WIRED registry must have its
``<name>.environment.json`` next to it. PRODUCES maps each wired script to the
paths its results take (the ``--out`` names used in the runbooks and docs);
PRE_CONTRACT lists, one by one, the results committed before the sidecar
contract existed (3fa9292, 2026-09-28 23:40 -05), whose environment was never
recorded and cannot be reconstructed. Nothing is exempt implicitly: a stale
entry (the file gained a sidecar, moved or was deleted) fails too.
"""

import fnmatch
import subprocess
from pathlib import Path

from dermssl import run_environment
from test_result_writers_environment import WIRED

ROOT = Path(__file__).resolve().parents[1]
SUFFIX = run_environment.SIDECAR_SUFFIX

# wired script -> fnmatch patterns (``*`` crosses directories) of the results it writes.
PRODUCES = {
    "plan-segmentacion/comun/evaluacion/compare_fst_manifests.py": ("plan-segmentacion/*/resultados/*/comparison*.json",),
    "plan-segmentacion/comun/evaluacion/evaluate_segmentation.py": (
        "*/evaluacion*/summary.json", "*/evaluaciones/*/summary.json", "*/pisos/*_summary.json"),
    "plan-segmentacion/comun/evaluacion/fst_pseudo_manifest.py": ("*/fst_pseudo_manifest.json",),
    "scripts/c1_cross_backbone.py": ("*/c1_cross_backbone.json",),
    "scripts/c1_final_eval.py": ("resultados/clasificacion/v7/final_eval.json", "resultados/clasificacion/v7/*/final_eval.json"),
    "scripts/c1_near_dup_embed.py": ("*/near_dup_embed*.json",),
    "scripts/c1_predict_masks.py": ("*/c1_pseudo_masks_manifest.json",),
    "scripts/c1_reevaluate_crop_gate.py": ("*/crop_regate_report.json",),
    "scripts/c1_run_grid.py": ("*/c1_bootstrap.json", "*/c1_cells.json"),
    "scripts/c1_select_variant.py": ("*/selection/wave*.json",),
    "scripts/c1_verify_bank_reproduction.py": ("*/bank_reproduction*.json",),
    "scripts/cluster_frozen_bank.py": ("resultados/clasificacion/bank.json", "*/bank.json"),
    "scripts/diagnostico_fst_comparativa.py": ("*diagnostico_fst*.json",),
    "scripts/kaggle_circle_stage2.py": ("*circle_stage2_results*.json",),
    "scripts/kaggle_fst_manifest.py": ("*/fst_manifest.json",),
    "scripts/pad_ufes_eval.py": ("*/pad_ufes_eval.json",),
    "scripts/pad_ufes_fst_validation.py": ("*/pad_ufes_fst_validation.json",),
    "scripts/probe_convergence_check.py": ("*/probe_convergence.json",),
    "scripts/c1_thread_repro_check.py": ("*/c1_thread_repro.json",),
    "scripts/release_parity.py": ("*/parity.json",),
    "scripts/run_segnorm_baseline.py": ("*/provenance.json",),
    "scripts/s4_dino_saliency.py": ("*/saliency_run.json", "*_saliency/run.json"),
    "scripts/s4_predict_pool.py": ("*/pool/run.json", "*/pseudo_a2/run.json"),
    "scripts/s4_unet_train.py": ("plan-segmentacion/camino-b-cnn/resultados/*-s4-*/run.json",),
    "scripts/s_diag_predict.py": ("*/sdiag/runs/*/*/run.json",),
    "scripts/seg_paired_bootstrap.py": ("*/paired_vs_*.json",),
    "scripts/t2_luminance_dice.py": ("*/t2_luminance_dice.json",),
    "scripts/t3_c1_luminance.py": ("*/t3_c1_luminance.json",),
    "scripts/t4_repro_compare.py": ("resultados/clasificacion/t4/diagnostico_reproduccion/*.json",),
    "scripts/t4_sog_run.py": ("*/t4_holdout_grid.json", "*/reproduction_failed.json"),
    "scripts/t4_sog_val.py": ("*/t4_selection.json", "*/t4_final_eval*.json"),
    "scripts/t6_luminance_threshold.py": ("*/t6_luminance_threshold.json",),
    "scripts/t7_compare.py": ("*/t7_compare.json",),
    "scripts/t7_r0_tercile_check.py": ("*/t7_r0_tercile_check.json",),
    "scripts/v71_extra_contrasts.py": ("*/v71/extra_contrasts.json",),
    "scripts/v71_run.py": ("*/v71/holdout_grid.json", "*/v71/pretest.json", "*/units/*/result.json"),
    "scripts/v71_val.py": ("*/v71/selection.json", "*/v71_final_eval*.json"),
    "scripts/validar_fitzpatrick17k.py": ("*validacion_fitzpatrick17k*.json",),
}

_CONTRACT = "committed before the sidecar contract (3fa9292, 2026-09-28 23:40 -05); "
_SEG = _CONTRACT + ("Camino B / S4 / S-diag evaluation, paired bootstrap or run record written on the pod; "
                    "its environment was never recorded and a rerun would be a different result")
_CLS = _CONTRACT + ("classification result (bank, C1, v7, v7.1) whose single evaluation or selection "
                    "must not be rerun to recover the environment it did not record")
_LUMINANCE = _CONTRACT + ("T1/T2/T3/T6 result registered in docs/27-31 and read once; "
                         "produced by code that put no environment next to it")

# result -> why it has no sidecar. One entry per file, never a pattern.
PRE_CONTRACT = {
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-20-piloto/evaluacion_ham10000/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-20-piloto/evaluacion_isic2018/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-final/baselines_v2/evaluacion_ham10000_rnorm/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-final/baselines_v2/evaluacion_ham10000_xnorm/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-final/baselines_v2/evaluacion_isic2018_rnorm/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-final/baselines_v2/evaluacion_isic2018_xnorm/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-final/evaluacion_ham10000/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-final/evaluacion_isic2018/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-v2/evaluaciones/final_ham10000_tta/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-v2/evaluaciones/final_isic2018_tta/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-v2/evaluaciones/v2_ham10000/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-v2/evaluaciones/v2_ham10000_tta/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-v2/evaluaciones/v2_isic2018/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-v2/evaluaciones/v2_isic2018_tta/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-v2/pisos/rnorm_ham10000_summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-v2/pisos/rnorm_isic2018_summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-v2/pisos/xnorm_ham10000_summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-v2/pisos/xnorm_isic2018_summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-fst-a2/comparison_vs_published.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-fst-hair/comparison.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a1/evaluacion_ham10000/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a1/evaluacion_isic2018/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a1/paired_vs_a2_ham10000.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a1/paired_vs_a2_isic2018.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a1/paired_vs_student_ham10000.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a1/paired_vs_student_isic2018.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a1/run.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a1/saliency_run.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a2-astar/evaluacion_ham10000/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a2-astar/evaluacion_isic2018/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a2-astar/paired_vs_a2_public_ham10000.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a2-astar/paired_vs_a2_public_isic2018.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a2-astar/paired_vs_student_ham10000.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a2-astar/paired_vs_student_isic2018.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a2-astar/run.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a2/evaluacion_ham10000/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a2/evaluacion_isic2018/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a2/paired_vs_student_ham10000.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a2/paired_vs_student_isic2018.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a2/run.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_ham10000_astar_kmeans/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_ham10000_astar_pca448/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_ham10000_astar_primary/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_ham10000_bstar_kmeans/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_ham10000_bstar_pca448/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_ham10000_bstar_primary/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_ham10000_public_kmeans/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_ham10000_public_pca448/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_ham10000_public_primary/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_isic2018_astar_kmeans/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_isic2018_astar_pca448/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_isic2018_astar_primary/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_isic2018_bstar_kmeans/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_isic2018_bstar_pca448/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_isic2018_bstar_primary/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_isic2018_public_kmeans/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_isic2018_public_pca448/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-sdiag/evaluacion_isic2018_public_primary/summary.json": _SEG,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-t2-luminance-dice/t2_luminance_dice.json": _LUMINANCE,
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-29-t6-luminance-threshold/t6_luminance_threshold.json": _LUMINANCE,
    "resultados/clasificacion/bank.json": _CLS,
    "resultados/clasificacion/c1/c1_bootstrap.json": _CLS,
    "resultados/clasificacion/c1/c1_cells.json": _CLS,
    "resultados/clasificacion/c1/c1_cross_backbone.json": _CLS,
    "resultados/clasificacion/c1_v7/a2/c1_bootstrap.json": _CLS,
    "resultados/clasificacion/c1_v7/a2/c1_cells.json": _CLS,
    "resultados/clasificacion/c1_v7/student_r3/c1_bootstrap.json": _CLS,
    "resultados/clasificacion/c1_v7/student_r3/c1_cells.json": _CLS,
    "resultados/clasificacion/c1_v7/t3_luminance/t3_c1_luminance.json": _LUMINANCE,
    "resultados/clasificacion/v7/final_eval.json": _CLS,
    "resultados/clasificacion/v7/s43/final_eval.json": _CLS,
    "resultados/clasificacion/v71/holdout_grid.json": _CLS,
    "resultados/clasificacion/v71/pretest.json": _CLS,
    "resultados/clasificacion/v71/selection.json": _CLS,
    "resultados/clasificacion/v71/v71_final_eval.json": _CLS,
    "resultados/fst/pad_ufes_t1/pad_ufes_fst_validation.json": _LUMINANCE,
}


def _tracked_json() -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.json"], capture_output=True, text=True, check=True)
    return out.stdout.split()


def _producers(rel: str) -> list[str]:
    return [script for script, patterns in PRODUCES.items() if any(fnmatch.fnmatch(rel, p) for p in patterns)]


def _results_of_wired_scripts() -> dict[str, list[str]]:
    return {rel: found for rel in _tracked_json() if not rel.endswith(SUFFIX) and (found := _producers(rel))}


def _sidecar_rel(rel: str) -> str:
    return str(run_environment.sidecar_path(Path(rel)))


def test_every_wired_script_says_which_results_it_writes():
    assert set(PRODUCES) == set(WIRED), set(PRODUCES) ^ set(WIRED)
    assert all(PRODUCES.values())


def test_every_committed_result_of_a_wired_script_has_its_environment_sidecar():
    tracked = set(_tracked_json())
    missing = [f"{rel} <- {', '.join(found)}" for rel, found in sorted(_results_of_wired_scripts().items())
               if _sidecar_rel(rel) not in tracked and rel not in PRE_CONTRACT]
    assert not missing, "results committed without <name>.environment.json:\n" + "\n".join(missing)


def test_every_pre_contract_exemption_is_still_true_and_has_a_reason():
    tracked, results = set(_tracked_json()), _results_of_wired_scripts()
    stale = [rel for rel in PRE_CONTRACT if rel not in results or _sidecar_rel(rel) in tracked]
    assert not stale, f"exemptions that no longer apply (gone, not a wired result, or now with a sidecar): {stale}"
    assert all(reason.strip() for reason in PRE_CONTRACT.values())


def test_no_committed_sidecar_is_orphaned_stale_or_misnamed():
    sidecars = [rel for rel in _tracked_json() if rel.endswith(SUFFIX)]
    assert sidecars, "the repository holds sidecars; finding none means the scan is broken"
    problems = [p for rel in sidecars for p in run_environment.sidecar_problems(ROOT / rel)]
    assert not problems, problems
