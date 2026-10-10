## ADDED Requirements

### Requirement: Traceable thesis figure inventory
The system SHALL maintain an inventory of planned and existing thesis figures with objective, placement, experiment/model/dataset/partition, variables and units, source artifact and revision, evaluation/resampling unit, evidence status, and missing inputs. Evidence status SHALL use only `Documented figures`, `Verified artifacts`, `Additional calculation`, `New experiment`, or `Pending`.

#### Scenario: Inventory distinguishes evidence states
- **WHEN** a figure has aggregate values but no verified source artifact
- **THEN** the inventory marks it `Documented figures` and does not claim the figure is reproducible

#### Scenario: Figure requires unavailable experiment
- **WHEN** a figure requires an experiment or source data that does not exist
- **THEN** the inventory marks it `New experiment` or `Pending` and records the missing input without generating values

### Requirement: Shared thesis figure style
The system SHALL provide an isolated style module with the specified colors, Nimbus Mono PS/Courier New font preference, 11 pt labels, 12 pt panel headings, specified line widths, and 17.3 cm figure width. It SHALL export editable-text SVG, embedded-font vector PDF, and 300 dpi PNG without tight-cropping away the intended physical dimensions. Existing `beautiful.py` and previously delivered figures SHALL remain unchanged.

#### Scenario: Export uses approved dimensions and formats
- **WHEN** a figure is exported through the shared style module
- **THEN** SVG, PDF, and PNG derive from the same figure at 17.3 cm width, with editable SVG text and embedded PDF fonts

#### Scenario: Required font is unavailable
- **WHEN** neither Courier New nor Nimbus Mono PS resolves to an installed font
- **THEN** export fails with a clear error instead of silently substituting another family

### Requirement: C1 macro-F1 figures
The system SHALL generate one horizontal-bar C1 figure for each of HAM10000, ISIC 2019, and ISIC 2020, comparing the selected B71 result with the eight available frozen baselines for that dataset. Each figure SHALL show macro-F1, support, actual protocol/partition, and source-reported individual 95% intervals when available. It SHALL keep fine-tuning results separate and SHALL state that individual intervals are not paired-difference intervals.

#### Scenario: HAM10000 comparison uses aligned E3b artifacts
- **WHEN** the HAM10000 C1 figure is built
- **THEN** B71-TTA and the eight frozen baselines come from the canonical E3b re-evaluation on the same v6/v7 validation partition and each value matches `e3b_bs64/e3b_summary.json`

#### Scenario: ISIC comparisons disclose their source protocols
- **WHEN** an ISIC C1 figure is built
- **THEN** B71-original and each frozen-baseline value comes from its committed JSON, and the caption states the dataset support and recorded transductive/protocol limitations

### Requirement: C2 paired macro-F1 forest figures
The system SHALL generate C2 forest figures only from committed paired contrast records. Each plotted point and interval SHALL preserve stored direction and source endpoints, identify the comparison set, support, resampling unit and count, and show zero reference. The figure notes SHALL disclose that no multiplicity correction is applied. Unavailable paired baselines SHALL not be reconstructed from aggregate metrics.

#### Scenario: HAM10000 paired comparisons
- **WHEN** the HAM10000 C2 figure is built
- **THEN** it contains the eight canonical E3b B71-TTA versus frozen-baseline macro-F1 contrasts, grouped by `lesion_id`, with 1,000 resamples and seed 20260923

#### Scenario: ISIC paired public controls
- **WHEN** an ISIC C2 figure is built
- **THEN** it uses only the canonical committed B71-original versus public-B71 contrast, identifies `image` for ISIC 2019 or `patient_id` for ISIC 2020, and does not imply paired comparisons with the eight classical baselines

### Requirement: Reproducible figure artifacts
Each C1/C2 figure SHALL have a reproducible script, SVG/PDF/PNG exports, a Spanish caption/note outside the plot canvas, and traceability data listing exact source files and hashes. Tests SHALL compare plotted data to source artifacts and verify export dimensions, font behavior, and format outputs.

#### Scenario: Source values change
- **WHEN** a committed source value or paired contrast changes
- **THEN** tests or the build validation detect a mismatch between plotted data and the source artifact

#### Scenario: Figure is regenerated
- **WHEN** a figure build succeeds
- **THEN** all three formats are written from the same figure and the inventory/traceability metadata names the source inputs
