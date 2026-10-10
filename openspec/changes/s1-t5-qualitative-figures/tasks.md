## 1. Shared manifest safeguards
- [x] 1.1 Add RED tests for manifest shape, binary masks, and traversal rejection.
- [x] 1.2 Implement shared safe path resolution, image/mask readers, and SHA-256 verification.

## 2. S1 segmentation comparison
- [x] 2.1 Add temporary-fixture tests for all quality levels, resolution, and Dice reconciliation.
- [x] 2.2 Implement the five-column S1 builder and traceability sidecars.
- [x] 2.3 Verify missing manifests fail closed; keep synthetic tests/figures out of repository outputs.
- [ ] 2.4 Generate and inspect S1 only after real pod manifest and files are transferred.

## 3. T5 tone pairs
- [x] 3.1 Add temporary-fixture tests for pair hashes, dimensions, mask protection, and derived L* shift.
- [x] 3.2 Implement the before/after/mask builder and traceability sidecars.
- [x] 3.3 Verify missing manifests fail closed; keep synthetic tests/figures out of repository outputs.
- [ ] 3.4 Generate and inspect T5 only after real pod manifest and files are transferred.

## 4. Verification and delivery
- [x] 4.1 Run all figure suites (183 passed), strict checks, and OpenSpec validation.
- [x] 4.2 Update the inventory with pending real inputs and verified transform semantics.
- [x] 4.3 Commit scripts/tests/docs atomically; do not commit synthetic exports.
