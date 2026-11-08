## ADDED Requirements

### Requirement: Complete proposed classifier
The release SHALL identify MoCo-v3-style quota12+4/CLS4 as the proposed method, bind seed42 and its source SHA, and contain the shared encoder, both trained readers, preprocessing, row-L2 normalization, classes and provenance.

#### Scenario: Native ontology outputs
- **WHEN** a complete model is loaded
- **THEN** HAM output SHALL have its seven declared classes and PAD output its six declared classes
- **AND** neither output SHALL silently remap labels, add D4, prior adjustment or StandardScaler

### Requirement: Development-only reader fitting
Final readers SHALL use only the sealed allowed pad_adapt/ham_dev manifests and development labels, with grouped three-fold repeated C selection and explicit scope distinct from OOF evaluation.

#### Scenario: Forbidden or misaligned input
- **WHEN** a role, hash, ID order, group, label order, feature dimension or finite-value check fails
- **THEN** fitting/export SHALL stop without using a replacement cohort

### Requirement: Portable parity
All forms SHALL preserve the same FP32 inference definition and pass maximum absolute probability error at most 1e-4, normalized finite probabilities, class order and parity-fixture argmax agreement.

#### Scenario: Independent verification chain
- **WHEN** publication is attempted
- **THEN** native Hub, plain PyTorch, safe checkpoint reload, ONNX CPU, offline Docker, candidate HTTP and final HTTPS evidence SHALL be recorded
- **AND** BF16 research-versus-FP32 serving drift SHALL be described separately

### Requirement: Versioned service and rollback
V10 SHALL expose MOCO_HAM7 by default and MOCO_PAD6 explicitly while preserving legacy A71/B71 identities and protecting V9 rollback.

#### Scenario: Promotion gate
- **WHEN** the candidate fails parity, resource, identity or readiness checks
- **THEN** production SHALL remain on or revert to V9 without deleting its image/models

### Requirement: Evidence-based release
The release SHALL point to tested committed source, provide PyTorch/ONNX/Docker usage and hashes, and generate lesion tables from aligned stored OOF evidence rather than hand-entered benchmark claims.

#### Scenario: Scientific scope
- **WHEN** development results are displayed
- **THEN** three encoder seeds SHALL not be described as an ensemble or nine independent trainings
- **AND** PAD transduction, HAM-dev selection and unexecuted tonal E06-E08/T0/M1 SHALL remain explicit
