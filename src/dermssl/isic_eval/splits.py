"""The committed ISIC 2019 / ISIC 2020 evaluation splits, rebuilt and verified (docs/33 sections 2 and 5).

The split is not stored: it is rebuilt from the dataset files with the recipe the baselines used
(ISIC 2019: ``StratifiedKFold``; ISIC 2020: ``StratifiedGroupKFold`` by ``patient_id``; both 5 folds,
shuffle, seed 42, first fold = val). ``StratifiedGroupKFold(shuffle=True)`` changed in scikit-learn 1.8,
so a rebuilt split is only trusted after its val class supports and its val id digest equal the committed
ones; any difference raises :class:`SplitMismatchError` and nothing is evaluated.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from dermssl import c1_split_identity as si
from dermssl import v71_plan as plan

N_SPLITS = 5
SEED = 42
ISIC2019_CLASSES = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")
ISIC2019_DROPPED = ("UNK",)
ISIC2020_CLASSES = ("benign", "malignant")


class SplitMismatchError(RuntimeError):
    """The rebuilt val split is not the committed one."""


@dataclass(frozen=True)
class SplitExpectation:
    """What the committed split looks like: class order, val supports and val id digest."""

    name: str
    classes: tuple[str, ...]
    val_supports: tuple[int, ...]
    val_sha256: str


ISIC2019_EXPECTED = SplitExpectation(
    "isic2019", ISIC2019_CLASSES, (905, 2575, 665, 174, 525, 47, 50, 126),
    "8cde8a71ccf5bf2e477ffdab445e8f1afbc37a164ee0cd1aba93ef785cc0b57f")
ISIC2020_EXPECTED = SplitExpectation(
    "isic2020", ISIC2020_CLASSES, (6538, 122),
    "ec3f5408881aa4623493e95e2840f360bab3de300c6fc022a4b88e55d5a0567e")


@dataclass(frozen=True)
class EvalDataset:
    """Ids, labels, optional bootstrap groups, image paths and the train / val row indices."""

    name: str
    classes: tuple[str, ...]
    ids: tuple[str, ...]
    labels: np.ndarray
    groups: np.ndarray | None
    paths: dict[str, Path]
    train_idx: np.ndarray
    val_idx: np.ndarray
    identity: dict = field(default_factory=dict)

    def _take(self, idx: np.ndarray) -> list[str]:
        return [self.ids[i] for i in idx]

    @property
    def train_ids(self) -> list[str]:
        return self._take(self.train_idx)

    @property
    def val_ids(self) -> list[str]:
        return self._take(self.val_idx)

    @property
    def y_train(self) -> np.ndarray:
        return self.labels[self.train_idx]

    @property
    def y_val(self) -> np.ndarray:
        return self.labels[self.val_idx]

    @property
    def val_groups(self) -> np.ndarray | None:
        return None if self.groups is None else self.groups[self.val_idx]


def sorted_ids_sha256(ids) -> str:
    """sha256 of the sorted ids joined by newline (the form printed in docs/34)."""
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()


def _val_sha_matches(recorded: str, ids) -> str | None:
    """Which construction (newline between ids, or after each id) reproduces ``recorded``, else None."""
    return si.identify_digest(recorded, sorted(map(str, ids)))


def verify_split(ds: EvalDataset, expected: SplitExpectation) -> dict:
    """Refuse unless classes, val supports and val sha256 equal ``expected``; return the identity block."""
    import sklearn  # noqa: PLC0415

    supports = tuple(int(c) for c in np.bincount(ds.y_val, minlength=len(ds.classes)))
    problems = []
    if tuple(ds.classes) != tuple(expected.classes):
        problems.append(f"classes {list(ds.classes)} != {list(expected.classes)}")
    if supports != tuple(expected.val_supports):
        problems.append(f"val supports {list(supports)} != committed {list(expected.val_supports)}")
    construction = _val_sha_matches(expected.val_sha256, ds.val_ids)
    if construction is None:
        problems.append(f"val sha256 {sorted_ids_sha256(ds.val_ids)[:12]} != committed {expected.val_sha256[:12]}")
    if problems:
        raise SplitMismatchError(
            f"{expected.name}: the rebuilt split is not the committed one ({'; '.join(problems)}). "
            f"scikit-learn {sklearn.__version__}; StratifiedGroupKFold only reproduces with <= 1.7.2")
    return {"sklearn_version": sklearn.__version__, "n_train": int(len(ds.train_idx)),
            "n_val": int(len(ds.val_idx)), "val_supports": list(supports),
            "val_ids_sha256": expected.val_sha256, "val_sha_construction": construction,
            "train_ids_sha256": si.ids_sha256(sorted(ds.train_ids))}


def index_images(image_dir: Path) -> dict[str, Path]:
    """Every ``*.jpg`` under ``image_dir`` by file stem (the baselines' lookup)."""
    return {p.stem: p for p in Path(image_dir).rglob("*.jpg")}


def _fold(splitter, ids, y, groups=None) -> tuple[np.ndarray, np.ndarray]:
    return next(splitter.split(ids, y, groups))


def _build(name, classes, ids, y, groups, paths, splitter, expected) -> EvalDataset:
    train_idx, val_idx = _fold(splitter, ids, y, groups)
    ds = EvalDataset(name, tuple(classes), tuple(ids), np.asarray(y, dtype=np.int64), groups,
                     {i: paths[i] for i in ids}, train_idx, val_idx)
    if groups is not None and set(groups[train_idx]) & set(groups[val_idx]):
        raise SplitMismatchError(f"{name}: patients in both train and val")
    identity = verify_split(ds, expected)
    return replace(ds, identity=identity)


def load_isic2019(csv_path: Path, image_dir: Path, expected: SplitExpectation = ISIC2019_EXPECTED) -> EvalDataset:
    """One-hot ground truth (``UNK`` dropped), images by stem, ``StratifiedKFold`` first fold = val."""
    import pandas as pd  # noqa: PLC0415
    from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

    paths = index_images(image_dir)
    meta = pd.read_csv(csv_path)
    id_col = meta.columns[0]
    meta = meta[meta[id_col].astype(str).isin(paths)].reset_index(drop=True)
    onehot = meta.drop(columns=[id_col] + [c for c in ISIC2019_DROPPED if c in meta.columns])
    ids = meta[id_col].astype(str).tolist()
    y = onehot.to_numpy().argmax(axis=1)  # same first-max rule as idxmax
    splitter = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    return _build("isic2019", list(onehot.columns), ids, y, None, paths, splitter, expected)


def load_isic2020(csv_path: Path, image_dir: Path, expected: SplitExpectation = ISIC2020_EXPECTED) -> EvalDataset:
    """``train-metadata.csv`` (isic_id, patient_id, target), ``StratifiedGroupKFold`` by patient, first fold = val."""
    import pandas as pd  # noqa: PLC0415
    from sklearn.model_selection import StratifiedGroupKFold  # noqa: PLC0415

    paths = index_images(image_dir)
    meta = pd.read_csv(csv_path)
    meta = meta[meta["isic_id"].astype(str).isin(paths)].reset_index(drop=True)
    ids = meta["isic_id"].astype(str).tolist()
    y = meta["target"].to_numpy().astype(np.int64)
    groups = meta["patient_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED)
    return _build("isic2020", expected.classes, ids, y, groups, paths, splitter, expected)


LOADERS = {"isic2019": (load_isic2019, "ISIC_2019_Training_GroundTruth.csv"),
           "isic2020": (load_isic2020, "train-metadata.csv")}


def limited(ds: EvalDataset, n: int | None) -> EvalDataset:
    """Smoke subset: ``n`` train and ``n`` val rows taken round-robin over the classes (after verification)."""
    if n is None:
        return ds
    picked = []
    for idx in (ds.train_idx, ds.val_idx):
        keep = set(plan.limited_ids(idx.tolist(), ds.labels[idx].tolist(), n))
        picked.append(np.array([i for i in idx if i in keep], dtype=np.int64))
    return replace(ds, train_idx=picked[0], val_idx=picked[1])
