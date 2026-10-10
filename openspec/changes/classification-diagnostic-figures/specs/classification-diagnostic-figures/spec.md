## ADDED Requirements

### Requirement: Melanoma discrimination curves
The system SHALL build C5 precision–recall and ROC curves from aligned per-image labels and probabilities only. It SHALL verify source hashes, label identity, class order, and IDs against committed metadata; define melanoma as positive; calculate AP and ROC AUC from individual scores; show observed partition prevalence as the PR baseline and the chance diagonal on ROC; and state that evaluation thresholds were not tuned. Each panel SHALL contain at most three model curves and SHALL mark unavailable comparisons as Pending rather than reconstructing them from aggregates.

#### Scenario: Probability archive is verified
- **WHEN** a C5 panel is built from a probability archive
- **THEN** every plotted score derives from per-image probabilities aligned to the verified labels and source metadata

#### Scenario: Individual probabilities are unavailable
- **WHEN** a model has only aggregate metrics or an archive without verified per-image alignment
- **THEN** the model is omitted and the inventory marks its curve Pending

### Requirement: Confidence reliability diagrams
C6 SHALL define confidence as the maximum class probability and correctness as argmax matching the true class. It SHALL use 15 equal-width bins with intervals `(lo, hi]` and first interval `[0, hi]`, show mean confidence and accuracy for each nonempty bin, and draw the perfect-calibration diagonal. It SHALL include an aligned per-bin support histogram on the same confidence axis, omit empty bins, and use hollow markers for bins with `n < 10` (threshold declared in the note). It SHALL recompute ECE using those same bins and include ECE only when the recomputed value matches the source JSON within stated numerical tolerance. It SHALL NOT reconstruct a reliability curve from an aggregate ECE.

#### Scenario: Confidence bins are plotted
- **WHEN** C6 receives aligned per-image labels and probabilities
- **THEN** each nonempty bin reports its support, mean confidence, and empirical accuracy under the declared bin convention

#### Scenario: Published ECE differs
- **WHEN** the recomputed ECE does not match the source JSON
- **THEN** the plot omits the ECE comparison and the note records the discrepancy

### Requirement: Bounded F4 ablation claims
C7 SHALL produce separate figures for A71 versus B71 and B71 versus B71+F4. The first figure SHALL describe A71/B71 as a bundled protocol contrast because extraction and probe also differ from tone. The second SHALL use stored paired F1 contrasts and available seed variability, define F4 as crop plus descriptors, and SHALL NOT attribute the combined effect to crop or descriptors individually.

#### Scenario: Combined F4 contrast is shown
- **WHEN** paired B71 and B71+F4 artifacts are available
- **THEN** C7 plots the verified contrast and interval while identifying the combined intervention and its exploratory status

#### Scenario: Isolated F4 components are requested
- **WHEN** no crop-only or descriptor-only experiment exists
- **THEN** those effects remain Pending and no component-specific value or interval is generated

### Requirement: Traceable C5–C7 exports
Each C5–C7 figure SHALL have SVG/PDF/PNG exports from the shared style module and Spanish caption/note plus traceability sidecars containing exact source paths and hashes, evaluation split, unit, metric definition, and limitations. Tests SHALL verify source fidelity, calculations, and output artifacts.

#### Scenario: Source artifacts fail validation
- **WHEN** a source digest, partition alignment, or required metadata does not match
- **THEN** figure generation aborts without presenting inferred or simulated results
