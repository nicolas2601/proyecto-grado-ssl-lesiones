## ADDED Requirements

### Requirement: Per-class classification metrics
C4 SHALL derive precision, sensitivity, and F1 for B71 and A71 from their committed count matrices, with rows as true class and columns as predicted class. It SHALL verify class order and per-class support match within each dataset, report support, and omit class-level confidence intervals unless a source provides them. It SHALL NOT compare HAM10000 matrices evaluated on different partitions.

#### Scenario: Matched model matrices are loaded
- **WHEN** B71 and A71 confusion matrices are loaded for one dataset
- **THEN** class order, count totals, and per-class support match before metrics are plotted

#### Scenario: HAM baseline has different support
- **WHEN** a candidate HAM10000 baseline uses a different partition or support
- **THEN** C4 excludes it from direct comparison and records the partition limitation

### Requirement: ITA geometry and intervals
T1 SHALL explain the ITA formula and its CIELAB axes as a conceptual schematic without experimental observations. T2 SHALL generate group intervals from `FST_THRESHOLDS` and the strict-greater-than rule in `fst_bucket`; equality at each cut point SHALL map to the darker group. Both artifacts SHALL identify I–VI as estimator groups, not clinically validated Fitzpatrick types.

#### Scenario: Threshold boundary is rendered
- **WHEN** a threshold value is classified
- **THEN** T2 assigns it to the darker interval and keeps invalid or abstained estimates outside the groups

#### Scenario: ITA schematic is exported
- **WHEN** T1 is built
- **THEN** the formula, `b* ≥ 5` reporting guard, and conceptual-only status are visible and traceable to source code

### Requirement: Continuous ITA distribution
T4 SHALL plot only finite `roi_ita` values whose source `roi_status` is `ok` for HAM10000, ISIC 2019, and ISIC 2020. It SHALL report valid, total, and abstained counts, use the shared verified threshold intervals, and identify the perilesional ROI and dataset for each violin.

#### Scenario: Invalid ROI estimates are present
- **WHEN** an E5 row has a non-`ok` ROI status
- **THEN** its ITA value is excluded from the violin and counted as an abstention

#### Scenario: Threshold metadata differs
- **WHEN** source E5 files use different thresholds or invalid row/value combinations
- **THEN** T4 generation aborts rather than combining inconsistent distributions

### Requirement: Traceable exports
C4, T1, T2, and T4 SHALL include Spanish captions or table captions, methodological notes, and source traceability. Figures SHALL export editable SVG, embedded-font PDF, and 300 dpi PNG through the shared thesis style. Tests SHALL verify calculations, boundaries, source alignment, and generated artifacts.

#### Scenario: Source validation fails
- **WHEN** a source hash, matrix invariant, threshold, or ROI invariant fails
- **THEN** the corresponding artifact is not generated
