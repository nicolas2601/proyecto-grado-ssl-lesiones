# Design: Per-class and ITA figures

## Decisions
- C4 reads only committed C3 count matrices. Use B71 and A71, the matched model pair available on each dataset's same split. On HAM10000, do not add frozen DINOv2: its matrix has n=2,014 while B71/A71 use n=2,004. Compare class metrics only within a matched dataset/split. Calculate per-class precision = TP/(TP+FP), sensitivity = TP/(TP+FN), and F1 = 2TP/(2TP+FP+FN); zero predicted support maps precision to 0, consistent with `src/dermssl/bench/metrics.py`. Do not invent per-class intervals.
- T1 is a schematic of the vector `(b*, L*−50)` and `ITA = atan((L*−50)/b*) × 180/π`; it contains no observations and states the `b* ≥ 5` reporting guard. It is not a clinical validation.
- T2 generates intervals from `FST_THRESHOLDS` and `fst_bucket`; exact cut points belong to the darker group. Show open-ended outer groups rather than implying unattainable endpoints. Keep abstentions outside groups and state I–VI are estimator categories, not validated clinical phototypes.
- T4 uses `roi_ita` for `roi_status == "ok"` from HAM10000/ISIC 2019/ISIC 2020 E5 JSON. Show one neutral violin per dataset over the shared Groh intervals; state valid n, total n, and abstentions. No values from invalid estimates enter the violins.
- Use `thesis_style.py`, fixed 17.3 cm width, Spanish captions/notes, and source hashes. Use fresh C4/T4 stems; never change existing figures or `beautiful.py`.

## Trade-offs
- Restricting C4 to B71/A71 avoids an unfair HAM10000 comparison across different partitions, but omits other baselines from this figure.
- T4 preserves the existing question and ROI source while separating continuous distributions from bucket counts; abstentions are described, not plotted as ITA values.

## Risks and rollback
- Abort generation when class order, count totals, split support, threshold values, or ROI status/value invariants disagree with source records.
- All artifacts use new paths; rollback removes only this change's generated artifacts and scripts/tests.
