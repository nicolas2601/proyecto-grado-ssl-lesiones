# Proposal: dermoscope field ROI, honest evaluation and live tracking for the label-free segmentation pipeline

## Why

The U-Net masks produced for ISIC 2019/2020 cover the whole bright circular field of the dermoscope and leave the lesion out. Root cause: nothing in the pipeline knows where the dermoscope field ends. The pseudo-mask module keeps the single most central connected region with a minimum area but no maximum, and the sign orientation compares a border ring that is black on vignetted images, so "bright centre = lesion" wins by construction. The vignette ring (grey 0.05-0.20) also passes the ITA border threshold (0.02) and is counted as skin, which produces the measured 39 % of dark skin classified as bucket 1.

The evaluation reports one global mean and hides catastrophic masks; it has no percentiles, no coverage, no stratification by skin tone and no visual overlays. Training metrics are JSONL only; the cluster has no reachable TensorBoard or MLflow UI.

Constraint kept: the professor's rule (2026-09-18). Training never reads masks or labels. Masks are read only by the evaluation stage.

## What Changes

- ADDED `field_roi`: dermoscope field validity mask (circle fitted to the non-black bounding box, inward margin), computed once in the cache stage and stored as `field.npy`.
- MODIFIED `uncertainty_module`: candidates restricted to the field, orientation ring taken inside the field, maximum area fraction of the field (default 0.6), outside-field pixels labelled background.
- MODIFIED `gpu_augment`: multiplicative radial vignette augmentation for MoCo and student views, never applied to masks.
- MODIFIED `fitzpatrick.estimate_fst*`: optional explicit border mask; `fst_pseudo_manifest` passes the field so ITA uses field pixels only.
- MODIFIED `evaluate_segmentation`: percentiles, predicted and reference coverage, stratification by FST bucket from a manifest, overlay grids (worst, best, random), optional Weights & Biases logging.
- ADDED `wandb_logging`: optional sink for `JsonlLogger`, enabled by `WANDB_API_KEY` plus `--wandb-project`; no-op otherwise.
- MODIFIED `run_pipeline`: field flags, several evaluation sets, FST manifest stage, W&B environment propagation.
- ADDED presets `cluster-piloto.json` and `cluster-final.json` and `RUNBOOK_CLUSTER.md` for `~/work/proyectonp` on the UNABIA cluster.

## Impact

- `plan-segmentacion/comun/nucleo/{field_roi,gpu_augment,run_logging,wandb_logging}.py`
- `plan-segmentacion/comun/datos/prepare_cache.py`
- `plan-segmentacion/camino-b-cnn/pseudo-mascaras/{uncertainty_module,make_pseudo_masks}.py`
- `plan-segmentacion/comun/evaluacion/{evaluate_segmentation,fst_pseudo_manifest}.py`
- `plan-segmentacion/camino-b-cnn/entrenamiento/run_pipeline.py`, `plan-segmentacion/comun/configs/`
- `src/dermssl/fitzpatrick.py` (backwards compatible optional parameter)
- Tests under `tests/test_seg_*.py`; static rules in `tests/test_seg_static.py` must keep passing.
