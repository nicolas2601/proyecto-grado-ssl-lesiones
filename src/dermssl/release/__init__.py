"""Release package of the v7 classifier: fused probe, exports and preprocessing.

Pure modules (``fuse``, ``preprocess``, ``package``) import no torch; the
torch half lives in ``torch_export`` following the repo's ``torch_*`` rule.
"""
