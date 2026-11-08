## Why
Nico selected MoCo-v3-style DINOv2 ViT-B/14 with a 12+4 source/PAD batch quota and last-four CLS concatenation as the current proposed classifier. V5 stores research OOF readouts, not a complete deployment classifier. The current API and V9 bundles use another recipe and cannot be updated by swapping the backbone alone.

## What Changes
- Freeze seed 42 and fit separate, grouped-development-selected balanced logistic readers for HAM7 and PAD6.
- Package the full shared-backbone classifier in safe PyTorch, dual-output ONNX and offline Docker forms.
- Generate reproducible lesion-level tables and E01 documentation, preserving historical results and study limitations.
- Add a versioned MoCo API with an explicit HAM7 default, PAD6 opt-in and unchanged legacy A71/B71 identities.
- Verify native/PyTorch/ONNX/Docker/HTTP parity before a controlled V10 publication and production switch with V9 rollback.

## Impact
No additional SSL training, sealed-test access, tonal treatment experiment, collaborator message, frontend redesign, paid service or unrelated production change is included. Existing dirty V9 work is preserved and excluded from unrelated staging.
