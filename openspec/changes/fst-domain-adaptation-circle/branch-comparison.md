# Branch comparison: `origin/feature/v6-gpu-ssl-pretraining`

## Evidence

- Comparison base: `032bdd1a0b41dd3f7ce279be6554af546d73f3ca`.
- Compared paths: `src/dermssl/pretrain/`, `scripts/pretrain.py`,
  `src/dermssl/circle.py`, `src/dermssl/mask_manifest.py`, and their relevant
  tests.
- `git diff HEAD...origin/feature/v6-gpu-ssl-pretraining -- <compared paths>`
  returned no paths.
- `git log HEAD..origin/feature/v6-gpu-ssl-pretraining -- <compared paths>`
  returned no commits.

## Decision

No change was selectively integrated from the remote branch because it has no
delta in the SDD SSL/domain-adaptation scope. Its additional fine-tuning,
results, and documentation changes are outside this work unit and are not
evidence for an SSL-adaptation checkpoint.

## CP3 boundary

The reproducible run contract now records the candidate checkpoint, objective,
weights, frozen split hash, seed, GPU budget, and resume path. The experiment
owner must still choose the adaptation objective and promotion checkpoint before
any full GPU run; no training result is asserted by this comparison.

## Paired-transformation integration

The selected smoke objective is `simsiam` with
`original_transformed_consistency`: each original image must have a transformed
counterpart at the same relative path and the run records the weighted SSL loss
and the unweighted representation-consistency loss separately. The approved
`circle_lab_l_star_v1` transformation is materialized into a separate directory
only after its traceable lesion mask passes QA; it never modifies source images.

This establishes the implementation contract for task 3.3. A learned checkpoint
is not promoted here: selection remains validation-only and task 5.2 remains
the pending GPU validation on UNAB.
