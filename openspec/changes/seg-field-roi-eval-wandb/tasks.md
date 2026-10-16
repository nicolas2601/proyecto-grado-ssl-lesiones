# Tasks

Strict TDD per task (failing test first). Files at most 300 lines, functions at most 40 lines. Artifacts in English. No emojis.

## WU1 Field ROI, pseudo-mask fixes, vignette augmentation, ITA border

- [x] 1.1 `comun/nucleo/field_roi.py`: `field_validity_mask(rgb_uint8, black_threshold, margin_fraction) -> (valid bool HxW, stats)`; `compute_field_cache(images_memmap, out_memmap, chunk) -> stats`. Safe on all-black input (returns full-frame valid with `method="fallback_full"`).
- [x] 1.2 `comun/datos/prepare_cache.py` writes `field.npy` (N,H,W uint8) and `field_stats.json` next to `images.npy`; flags `--field-black-threshold`, `--field-margin-fraction`, `--no-field`.
- [x] 1.3 `uncertainty_module.py`: `saliency_to_pseudo_mask(..., field=None, max_area_fraction=0.6)`: candidates AND field; orientation ring inside the field; region area over field area above the cap -> sample invalid; pixels outside the field -> BACKGROUND. Stats add `excluded_by_area`.
- [x] 1.4 `make_pseudo_masks.py`: `--field PATH` (optional), `--max-area-fraction`.
- [x] 1.5 `gpu_augment.py`: `random_vignette(images, generator, probability, radius_range, strength_range, gamma_range)`; wired into `moco_augment` (p=0.3) and `student_augment` (p=0.3); masks untouched.
- [x] 1.6 `src/dermssl/fitzpatrick.py`: `estimate_fst(..., border_mask=None)` and `estimate_fst_from_mask_record(..., border_mask=None)`; border = luminance border OR border_mask. `fst_pseudo_manifest.py --field PATH` passes `~field` per image.

## WU2 Evaluation and live tracking

- [x] 2.1 `evaluate_segmentation.summarize`: add p5, p10, p25, p75, p90 and `worst_ids` (10 lowest Dice).
- [x] 2.2 Per-image `pred_coverage` and `ref_coverage` columns.
- [x] 2.3 `--fst-manifest CSV` (`image_id,fst_bucket`): summary per bucket with `min_n` (default 30), `SKIPPED` status below it, plus `gap_dice = max - min` across qualifying buckets.
- [x] 2.4 `--overlay-dir DIR --overlay-count N`: PNG grids `image | reference | prediction | overlay` for worst-N, best-N, random-N (seeded).
- [x] 2.5 `comun/nucleo/wandb_logging.py`: `WandbSink` (init/log/log_images/finish), enabled only when `WANDB_API_KEY` is set and a project is given; import guarded.
- [x] 2.6 `run_logging.JsonlLogger(path, echo, sinks=())` forwards each record to sinks. Wire `--wandb-project`, `--wandb-run-name` (default from output dir) into `train_moco.py`, `train_ccam.py`, `student_engine.py`/`train_student.py` on the main rank only.
- [x] 2.7 `evaluate_segmentation --wandb-project`: logs summary, per-bucket table, Dice histogram, overlay images.

## WU3 Pipeline wiring, presets, runbook

- [x] 3.1 `run_pipeline.py`: pass `--field <cache>/field.npy` and `max_area_fraction` to pseudo stages; `--eval-set NAME=IMAGES:MASKS` repeatable (keeps `--eval-image-dir/--eval-mask-dir`); `--fst-manifest` for evaluation; new stage `fst` after the last experimento de segmentación round; `--wandb-project` exported to child processes with `WANDB_RUN_GROUP=<workdir name>`.
- [x] 3.2 `comun/configs/cluster-piloto.json` and `cluster-final.json` with the fields above; no `<...>` placeholders.
- [x] 3.3 `plan-segmentacion/RUNBOOK_CLUSTER.md`: exact commands for `~/work/proyectonp` (venv with system site packages, data layout, ISIC 2018 Task 1 and HAM10000 mask downloads, smoke, pilot, final, evaluation on both sets, W&B links, nohup logs, resume).
- [x] 3.4 Update `tests/test_seg_run_pipeline.py` and `tests/test_seg_static.py` allow-lists if a new module legitimately opens images.

## Verification

- [x] V1 `ruff check` clean; `pytest tests/test_seg_*.py` green on CPU.
- [x] V2 `run_pipeline.py --smoke` completes on CPU with the field stage.
- [x] V3 Reviewer pass before commit.
