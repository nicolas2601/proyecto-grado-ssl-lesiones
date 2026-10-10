# Design: C1/C2 figure layer

## Decisions
- Keep `beautiful.py` and all previously delivered figures untouched. New C1/C2 scripts import only the new `thesis_style.py` module.
- Use the existing loaders in `dermssl.bench.rows` and `dermssl.bench.contrasts_load`; never scrape Markdown tables or type metric values into plotting code.
- C1 is descriptive: one horizontal-bar SVG/PDF/PNG per dataset, with B71 and eight frozen baselines. HAM10000 uses the aligned E3b canonical re-evaluation (B71-TTA and all eight baselines on v6/v7 validation, n=2,004); ISIC uses the recorded B71-original canonical cell and the eight existing frozen-baseline records, with partition/regime and n stated. C1 error bars are source-reported individual 95% intervals; they are not difference intervals and support no pairwise significance claim.
- C2 is inferential only for stored paired contrasts: (1) HAM10000 B71-TTA against eight frozen baselines from `e3b_bs64/e3b_summary.json`; (2) ISIC 2019 and ISIC 2020 B71-original against matched public B71 controls from canonical E1 contrast JSON. Classical ISIC baseline pairwise contrasts are omitted because their per-image paired predictions are not stored. Each panel/file states its comparison set and resampling unit. No multiplicity correction is claimed; source tables warn that the broader contrast family is unadjusted.
- All C2 estimates, endpoints, supports, units, and resampling counts come from source JSON. For E3b, bootstrap is 1,000 resamples, seed 20260923, grouped by `lesion_id`; canonical ISIC metadata identifies image or `patient_id`, and `scoring.py` defines 1,000 resamples with seed 20260923.
- Keep captions, notes, source paths, hashes, and limitations outside plot canvases in the figure inventory/traceability JSON.
- The new style uses white background; requested neutral/accent palette; reserved ITA colors are not used; 11 pt labels and ticks, 12 pt panel headings, no bold; thin specified strokes; SVG text remains editable; PDF uses embedded TrueType fonts; PNG is 300 dpi. Set width to 17.3 cm and save without `bbox_inches='tight'`.

## Trade-offs
- C1 spans several existing evaluation sources because the project does not have a unified benchmark partition. It labels each protocol and remains descriptive.
- C2 includes fewer comparisons than C1 where paired predictions are unavailable. This avoids manufacturing uncertainty from independent point estimates.
- This host initially lacked Courier New and Nimbus Mono PS. Nimbus Mono PS has now been installed from the `gsfonts` system package; exports must verify the resolved font and embedded PDF font before they are considered complete.

## Risks and rollback
- If source rows, support, or stored contrast direction fail validation, abort that figure export.
- Keep new stems separate; no prior figure is replaced.
- If font or physical dimensions fail export checks, stop delivery and fix the style module; do not silently fall back to a different font.
