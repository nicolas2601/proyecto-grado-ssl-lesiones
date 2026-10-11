# Proposal: modular v8 audit before new model training

## Why

The current tone comparison changes both the primary SimSiam pair and the additional
consistency objective. Segmentation errors and ROI selection errors remain coupled.
The latest consolidated branch already includes E5 perilesional estimation; reuse it.

## What changes

- Add a CPU observer/exporter of actual dataloader views and aligned pseudo-mask overlays.
- Add explicit mask encodings, exclusions, deterministic sampling and input/code hashes.
- Add JSONL-driven descriptive tables with a freshness check.
- Register staged label-free segmentation, ROI and domain experiments in docs/35.
- Keep all historical training behavior, checkpoints and reported metrics unchanged.

## Scope

This delivery implements W0 only. W1-W4 are designed, not executed.
Expert masks and diagnoses never enter SSL or segmentation training.
New model/threshold selection cannot use a previously published test as pristine.
No new scientific result is claimed from synthetic smoke validation.

