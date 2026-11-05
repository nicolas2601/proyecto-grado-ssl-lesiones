"""Bind diagnosis-free sampling arrays to exact ordered FIT image identities."""
import hashlib
import io
from pathlib import Path

import numpy as np


def load_sampling_plan(path, expected_sha256, images):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Sampling-plan digest changed")
    expected = np.asarray([Path(image).stem for image in images])
    if len(expected) != len(set(expected)):
        raise ValueError("Training images have duplicate identities")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != {"image_ids", "probabilities", "domains"}:
            raise ValueError("Sampling plans contain only image identities, weights and source domains")
        if not np.array_equal(archive["image_ids"], expected):
            raise ValueError("Sampling-plan identity order differs from the training loader")
        probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
        domains = archive["domains"]
    if (probabilities.shape != (len(expected),) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or probabilities.sum() <= 0):
        raise ValueError("Invalid FIT sampling probabilities")
    if domains.shape != probabilities.shape or domains.dtype.kind not in "iu" or not np.isin(domains, [0, 1]).all():
        raise ValueError("Invalid image-source domains")
    if not np.any(domains == 0) or any(probabilities[domains == value].sum() <= 0 for value in np.unique(domains)):
        raise ValueError("Every declared source domain needs positive sampling mass")
    return {"probabilities": probabilities,
            "domains": domains if np.any(domains == 1) else None}
