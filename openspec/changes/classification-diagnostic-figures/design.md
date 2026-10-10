# Design: Classification diagnostic figures C5–C7

## Decisions
- C5 reads only per-image NPZ predictions. Verify probability checksums, labels, class order, and IDs against committed metadata. Convert melanoma to the positive class. Show PR baseline as observed partition prevalence and ROC chance diagonal. Compute AP from recall increments and precision at each distinct score; compute ROC AUC from empirical curve trapezoids. Do not choose thresholds on evaluation data. Limit each panel to three model curves; omit or mark unavailable arms rather than reconstructing them.
- C6 defines confidence as maximum class probability and correctness as argmax matching the true label. Use 15 equal-width bins `(lo, hi]`, with the first bin `[0, hi]`; show empirical accuracy, mean confidence, and `n` in every nonempty bin. Recompute ECE from those same bins and include it only when it matches source JSON within numerical tolerance. Never infer a curve from an aggregate ECE.
- C7 uses two independent figures. A71 versus B71 is a bundled protocol contrast: extraction and probe differ as well as tone, so it cannot isolate tone. B71 versus B71+F4 uses the stored paired F1 contrast and seed variability; F4 means crop plus descriptors. Do not claim crop-only or descriptor-only effects.
- Use the existing thesis style module; keep Spanish captions, limitations, and hashes in sidecars. Existing figures and `beautiful.py` remain untouched.

## Trade-offs
- Showing fewer arms reduces clutter but leaves unavailable classical baselines explicitly pending.
- ECE consistency is a hard gate; datasets whose archive and published metric use different definitions remain eligible for a reliability diagram but not a recalculated ECE annotation.
- Seed-level robustness intervals describe observed runs and are not a substitute for a new experiment or a pooled inferential interval.

## Risks and rollback
- Abort when NPZ checksums, IDs, labels, class order, or source ECE do not match verified records.
- Use new output stems. Remove only this change's generated exports to roll back; no source experiment data are modified.
