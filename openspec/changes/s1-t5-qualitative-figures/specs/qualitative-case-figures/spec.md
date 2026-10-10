## ADDED Requirements

### Requirement: Verified S1 segmentation examples
S1 SHALL read cases from `resultados/insumos_pod/S1/manifest.json` and require at least one case for each level `alto`, `medio`, and `bajo`. Each case SHALL identify dataset and image ID, provide percentile criterion and both Dice values, and point to original, reference, student_r3, and S4-A2 images/masks. It SHALL verify all paths remain inside the manifest directory, IDs are unique per dataset, resolutions match, masks are binary, and mask-derived Dice matches manifest values within 1e-4. The figure SHALL show five columns: Original, Referencia, student_r3, S4-A2, and Contornos. It SHALL NOT omit low- or medium-quality cases to show only successful examples.

#### Scenario: A quality level is missing
- **WHEN** the manifest lacks at least one high, medium, or low case
- **THEN** S1 generation aborts without writing figure outputs

#### Scenario: Mask data do not match manifest
- **WHEN** a mask is non-binary, has a mismatched resolution, or yields Dice outside tolerance
- **THEN** S1 generation aborts before output creation

#### Scenario: Input path escapes manifest root
- **WHEN** any routed file is absolute or resolves outside the S1 input directory
- **THEN** S1 rejects the manifest and writes no figure

### Requirement: Verified T5 tone pairs
T5 SHALL read paired original/transformed images and lesion masks from `resultados/insumos_pod/T5/manifest.json`. It SHALL verify source, transformed, and mask SHA-256 values; matching image IDs and dimensions; binary mask values; shift and tercile metadata; and `shift_buckets` within the source-supported range. It SHALL derive signed pre-clipping `L_shift = -8*s` from `shift_buckets=s`, using the code-defined 8 L* step. If a manifest also supplies `L_shift`, it SHALL match the derived value. The figure SHALL show Original, Transformed, and mask panels, with `s`, derived `L_shift`, and tercile visible. It SHALL label mask semantics as `true=lesion` and state lesion pixels are protected.

#### Scenario: Shift field matches source code
- **WHEN** a tone-pair case has `shift_buckets=s`
- **THEN** T5 shows `L_shift=-8*s` as the requested pre-clipping change and confirms the mask/hash inputs

#### Scenario: Manifest shift conflicts with derived shift
- **WHEN** the supplied `L_shift` differs from `-8*s`
- **THEN** T5 aborts rather than displaying inconsistent transformation metadata

#### Scenario: Lesion-mask pixels are unchanged
- **WHEN** a pair is visualized
- **THEN** the same `true=lesion` mask is overlaid on both images and the note identifies lesion preservation

### Requirement: No simulated production exports
S1 and T5 SHALL abort without outputs when required manifests or real input files are missing. Synthetic fixtures MAY be used only within test temporary directories; they SHALL NOT create or replace repository result artifacts.

#### Scenario: Pod inputs have not been transferred
- **WHEN** the expected manifest is absent
- **THEN** the builder reports the missing manifest and creates no output directory or figure

#### Scenario: Tests use synthetic fixtures
- **WHEN** a test constructs temporary images and manifests
- **THEN** all fixture inputs and generated test outputs remain under pytest `tmp_path`

### Requirement: Traceable qualitative exports
Each S1/T5 figure SHALL export editable SVG, embedded-font PDF, and 300 dpi PNG with Spanish caption, methodological note, and traceability containing manifest and source hashes, image IDs, metrics/shift values, and input paths. Tests SHALL cover validation failures and successful generation exclusively in temporary directories.

#### Scenario: Manifest hash differs from an input
- **WHEN** any file digest fails verification
- **THEN** figure generation aborts before writing outputs
