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


def effective_k(requested_k: int, n_train: int, n_classes: int = 0) -> int:
    """Clip ``k`` to the labelled set size. A crash guard, nothing more.

    What we measured, and why ``k`` is deliberately *not* shrunk further
    ----------------------------------------------------------------------
    The Kaggle run of 2026-07-29 reported accuracy 0.663 next to macro-F1 0.181
    for the k-NN probe. Reproducing it with the real HAM10000 class proportions
    (648 labels, 7 classes, so ``df`` gets ~10 samples against ``nv``'s ~440)
    confirmed the mechanism: with ``k=20`` the probe only ever predicts **2 of 7
    classes**, because a class with 10 labelled samples cannot out-vote a class
    with 440.

    The tempting fix - shrink ``k`` - was measured and rejected:

    ==== ======== ========= ====== =================
    k    accuracy macro-F1  AUC    classes predicted
    ==== ======== ========= ====== =================
    20   0.639    0.124     0.738  2
    10   0.651    0.170     0.698  4
    7    0.639    0.183     0.674  5
    5    0.627    0.209     0.575  5
    3    0.615    0.202     0.563  7
    ==== ======== ========= ====== =================

    Lowering ``k`` buys a little macro-F1 and destroys AUC. It is a worse trade,
    not a fix.

    The real conclusion is about *reporting*, not about ``k``: under 58:1
    imbalance the argmax of an unweighted neighbour vote is dominated by the
    class prior, so **k-NN macro-F1 is not a meaningful headline number**. Read
    AUC for k-NN (rank based, prior independent), read macro-F1 from a probe that
    can weight classes - see :func:`logreg_predict_proba` with
    ``class_weight='balanced'`` and :func:`knn_balanced_predict_proba`.
    """
    if n_train <= 0:
        raise ValueError("k-NN probe needs at least one labelled training sample")
    return int(max(1, min(requested_k, n_train)))


def knn_predict_proba(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    n_classes: int,
    k: int = 20,
    temperature: float = 0.07,
    balanced: bool = False,
) -> np.ndarray:
    """Temperature-weighted cosine k-NN, as used to evaluate DINO/DINOv2.

    Each of the ``k`` nearest training neighbours votes for its class with
    weight ``exp(similarity / temperature)``. A low temperature makes the closest
    neighbour dominate; 0.07 is the value from the DINO paper.

    ``k`` is reduced by :func:`effective_k` for small or imbalanced label
    budgets. With ``balanced=True`` each vote is divided by how often its class
    appears in the labelled subset, which removes the majority-class prior; that
    is reported as a **separate** probe rather than applied silently, because it
    is a different estimator and not a bug fix.
    """
    x_train = l2_normalize(x_train)
    x_val = l2_normalize(x_val)
    y_train = np.asarray(y_train)
    k = effective_k(k, len(x_train), n_classes)

    class_weight = np.ones(n_classes, dtype=np.float64)
    if balanced:
        counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
        class_weight = 1.0 / np.maximum(counts, 1.0)  # unseen classes keep weight 1

    proba = np.zeros((len(x_val), n_classes), dtype=np.float64)
    for start in range(0, len(x_val), CHUNK):
        block = x_val[start : start + CHUNK]
        sims = block @ x_train.T  # cosine similarity, both sides normalised
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        top_sims = np.take_along_axis(sims, top, axis=1)
        weights = np.exp(top_sims / temperature)
        for row in range(len(block)):
            np.add.at(proba[start + row], y_train[top[row]], weights[row])
    return _counts_to_proba(proba * class_weight)


def knn_balanced_predict_proba(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    n_classes: int,
    k: int = 20,
    temperature: float = 0.07,
) -> np.ndarray:
    """Class-balanced k-NN: same neighbours, votes divided by class frequency.

    Reported alongside plain k-NN so the effect of removing the majority-class
    prior is visible instead of assumed.
    """
    return knn_predict_proba(
        x_train, y_train, x_val, n_classes, k=k, temperature=temperature, balanced=True
    )


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
    class_weight: str | None = None,
) -> np.ndarray:
    """Standard linear probe. Requires scikit-learn (present on Kaggle).

    Features are standardised first: logistic regression converges far better on
    zero-mean unit-variance inputs, and ViT feature scales vary a lot by layer.
    Columns absent from ``y_train`` are filled with zero probability so the
    output always has ``n_classes`` columns.

    ``class_weight='balanced'`` reweights the loss by inverse class frequency.
    This is the probe where macro-F1 *is* meaningful under heavy imbalance, since
    the estimator itself can be told not to follow the class prior - unlike an
    unweighted neighbour vote (see :func:`effective_k`).
    """
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("logreg probe needs scikit-learn: pip install scikit-learn") from exc

    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(
        max_iter=max_iter, C=c, random_state=seed, class_weight=class_weight
    )
    model.fit(scaler.transform(x_train), y_train)
    partial = model.predict_proba(scaler.transform(x_val))

    full = np.zeros((len(x_val), n_classes), dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        full[:, int(cls)] = partial[:, col]
    return full


def logreg_balanced_predict_proba(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    n_classes: int,
    max_iter: int = 2000,
    c: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Linear probe with inverse-frequency class weights.

    Reported next to the unweighted version rather than replacing it: whether the
    reweighting helps macro-F1 on these features is an empirical question, and
    the run answers it. We do not presume the answer.
    """
    return logreg_predict_proba(
        x_train, y_train, x_val, n_classes,
        max_iter=max_iter, c=c, seed=seed, class_weight="balanced",
    )


PROBES = {
    "knn": knn_predict_proba,
    "knn_balanced": knn_balanced_predict_proba,
    "prototype": prototype_predict_proba,
    "logreg": logreg_predict_proba,
    "logreg_balanced": logreg_balanced_predict_proba,
}

# Probes whose scores are similarity-derived rather than fitted probabilities.
# Their ECE is not meaningful: the softmax temperature is an arbitrary choice, not
# a calibrated one. Reported for completeness, never used as a calibration claim.
UNCALIBRATED_PROBES = ("knn", "knn_balanced", "prototype")

# Probes whose argmax follows the class prior under heavy imbalance, so their
# macro-F1 must not be quoted as a headline result. See effective_k() for the
# measurements behind this. AUC stays valid for all probes: it is rank based.
PRIOR_DOMINATED_PROBES = ("knn", "prototype")


def headline_metric(probe: str) -> str:
    """Which metric may be quoted as this probe's result."""
    return "auc" if probe in PRIOR_DOMINATED_PROBES else "macro_f1"


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
