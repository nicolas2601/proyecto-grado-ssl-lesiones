# Design: Segmentation and tone-pair figures

## Decisions
- Both builders load a JSON manifest plus image paths relative to that manifest's directory. Resolve paths and reject absolute paths, traversal, or symlinks escaping the input root. A missing manifest or file aborts before the output directory is created.
- S1 requires unique `(dataset, image_id)` rows and at least one case in each `nivel` (`alto`, `medio`, `bajo`). Every case must provide a non-empty percentile `criterio`, finite Dice values in [0,1], the same-resolution original and three binary masks, and exact per-image Dice matching the masks within 1e-4. All three strata appear; no hand-picked good-only subset. Render five columns: Original, Referencia, student_r3, S4-A2, Contornos. Masks are black/white; overlay contours use distinct colors and line styles.
- T5 uses `shift_buckets` from the transformation manifest as `s`. Source code defines `FST_SHIFT_L_STEP=8`; compute the signed requested shift `L_shift=-8*s` before clipping. Thus `s=2` means −16 L*, matching docs/22; no code/document discrepancy. Verify source, transformed, and mask SHA-256 values; require same IDs and resolution, finite bounded `s`, same dimensions, tercile metadata, and binary lesion masks. If present, manifest `L_shift` must equal the derived signed value.
- T5 shows Original, Transformed, and the same black/white lesion mask per case; image panels outline the protected lesion. Annotate `s` and derived `L_shift` as pre-clipping metadata, not measured post-clipping pixel delta. Label mask semantics `true=lesion` and state lesion pixels are preserved.
- Use `thesis_style.py`, 17.3 cm width, editable SVG/PDF/PNG, Spanish captions/notes, and SHA-256 traceability. Synthetic fixtures may be generated only inside test `tmp_path`; no synthetic outputs under repository result directories.

## Trade-offs
- Strict path, checksum, mask, resolution, and Dice checks can block an export when pod metadata is incomplete; this prevents plausible-looking but unverifiable examples.
- S1 and T5 scripts can be tested before pod data arrives, but production exports remain unavailable until real manifests and files are transferred.

## Risks and rollback
- Abort if required strata, paired IDs, mask semantics, Dice, hashes, dimensions, or shift metadata disagree.
- Outputs use fresh stems and are generated only from verified manifest inputs; rollback removes only this change's scripts, tests, and exports.
