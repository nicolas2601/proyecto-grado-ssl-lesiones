"""Frozen-backbone probes: how we evaluate without ever training the backbone.

Why this module exists
----------------------
The claim of the thesis is "no label ever touched the feature extractor". To
keep that claim true, evaluation must read the frozen features and fit only a
tiny classifier on top. Three probes, in increasing number of trained
parameters:

``knn_predict_proba``
    **Zero trained parameters.** Classify a validation image by the labels of
    its nearest training neighbours in feature space (cosine distance). This is
    the cleanest possible measure of representation quality: nothing is learned,
    so the number reflects the geometry of the features and nothing else.

``prototype_predict_proba``
    **Zero trained parameters.** One average feature vector per class; classify
    by closest average. Robust with very few labels, which is exactly the 1 %
    regime.

``logreg_predict_proba``
    **One linear layer.** Multinomial logistic regression, the standard "linear
    probe" of the SSL literature. Needs scikit-learn, imported lazily.

All three take features as plain numpy arrays, which is why the expensive part
(running images through the backbone) happens once and is cached.
"""

from __future__ import annotations

import numpy as np

CHUNK = 512  # rows of the validation set scored per matmul; bounds peak memory


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale every row to unit length so a dot product *is* cosine similarity."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def _counts_to_proba(counts: np.ndarray) -> np.ndarray:
    """Normalise per row; rows that sum to zero become a uniform distribution."""
    total = counts.sum(axis=1, keepdims=True)
    uniform = np.full_like(counts, 1.0 / counts.shape[1])
    return np.where(total > 0, counts / np.maximum(total, 1e-12), uniform)


def knn_predict_proba(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    n_classes: int,
    k: int = 20,
    temperature: float = 0.07,
) -> np.ndarray:
    """Temperature-weighted cosine k-NN, as used to evaluate DINO/DINOv2.

    Each of the ``k`` nearest training neighbours votes for its class with
    weight ``exp(similarity / temperature)``. A low temperature makes the
    closest neighbour dominate; 0.07 is the value from the DINO paper.

    ``k`` is clipped to the number of available training samples, so the 1 %
    regime does not crash.
    """
    x_train = l2_normalize(x_train)
    x_val = l2_normalize(x_val)
    y_train = np.asarray(y_train)
    if len(x_train) == 0:
        raise ValueError("k-NN probe needs at least one labelled training sample")
    k = int(min(k, len(x_train)))

    proba = np.zeros((len(x_val), n_classes), dtype=np.float64)
    for start in range(0, len(x_val), CHUNK):
        block = x_val[start : start + CHUNK]
        sims = block @ x_train.T  # cosine similarity, both sides normalised
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        top_sims = np.take_along_axis(sims, top, axis=1)
        weights = np.exp(top_sims / temperature)
        for row in range(len(block)):
            np.add.at(proba[start + row], y_train[top[row]], weights[row])
    return _counts_to_proba(proba)


def prototype_predict_proba(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    n_classes: int,
    temperature: float = 0.1,
) -> np.ndarray:
    """Nearest-class-mean classifier in cosine space.

    Classes absent from ``y_train`` get a zero prototype, which yields a
    similarity of 0 - never selected over a real class, never crashing.
    """
    x_train = l2_normalize(x_train)
    x_val = l2_normalize(x_val)
    y_train = np.asarray(y_train)

    protos = np.zeros((n_classes, x_train.shape[1]), dtype=np.float32)
    for cls in range(n_classes):
        mask = y_train == cls
        if mask.any():
            protos[cls] = x_train[mask].mean(axis=0)
    protos = l2_normalize(protos)

    logits = (x_val @ protos.T) / temperature
    logits -= logits.max(axis=1, keepdims=True)  # stabilise before exp
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def logreg_predict_proba(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    n_classes: int,
    max_iter: int = 2000,
    c: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Standard linear probe. Requires scikit-learn (present on Kaggle).

    Features are standardised first: logistic regression converges far better on
    zero-mean unit-variance inputs, and ViT feature scales vary a lot by layer.
    Columns absent from ``y_train`` are filled with zero probability so the
    output always has ``n_classes`` columns.
    """
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("logreg probe needs scikit-learn: pip install scikit-learn") from exc

    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=max_iter, C=c, random_state=seed)
    model.fit(scaler.transform(x_train), y_train)
    partial = model.predict_proba(scaler.transform(x_val))

    full = np.zeros((len(x_val), n_classes), dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        full[:, int(cls)] = partial[:, col]
    return full


PROBES = {
    "knn": knn_predict_proba,
    "prototype": prototype_predict_proba,
    "logreg": logreg_predict_proba,
}


def run_probe(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    n_classes: int,
    **kwargs,
) -> np.ndarray:
    """Dispatch by probe name so the notebook can loop over probes by string."""
    if name not in PROBES:
        raise ValueError(f"unknown probe {name!r}; expected one of {sorted(PROBES)}")
    return PROBES[name](x_train, y_train, x_val, n_classes, **kwargs)
