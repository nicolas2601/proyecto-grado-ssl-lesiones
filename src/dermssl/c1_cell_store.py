"""Per-cell storage for the C1 grid: atomic, resumable, and earned.

The pod died four times during v6. A 24-cell grid that loses everything on the
fourth crash never finishes, so each cell lands on disk as soon as it is done
and a relaunch skips what is already there.

The dangerous half is the skip
------------------------------
Skipping because a FILE EXISTS trusts the filename. A cell extracted before
the centre-crop fix has exactly the same name and features that are one row
off; reusing it would poison the grid with numbers that look fine. So a cell
is reused only when its manifest agrees on everything that could change what
the features MEAN -- the code revision, the masks, the split, the condition --
and when the features still hash to what the manifest recorded.

Each of those is a separate check because each fails differently: a stale
revision is "we fixed a bug since", different masks are "the segmenter
changed", a different split is "other images", and a bad hash is "this file
is not what we wrote".
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dermssl.feature_cache import _atomic_write, _digest
from dermssl.mask_conditions import CONDITIONS

#: Everything that changes what a cell's features mean. A cell saved without
#: all of these can never be safely reused, so it is refused at write time
#: rather than discovered to be unusable on the relaunch that needed it.
REQUIRED_PROVENANCE = (
    "code_revision",
    "masks_sha256",
    "c1_train_ids_sha256",
    "c1_val_ids_sha256",
)


@dataclass(frozen=True)
class LoadedCell:
    features: np.ndarray
    manifest: dict


class CellStore:
    """Directory of finished cells, one ``(backbone, condition)`` each."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, backbone: str, condition: str) -> str:
        safe = backbone.replace("/", "-")
        return f"cell__{safe}__{condition}"

    def features_path(self, backbone: str, condition: str) -> Path:
        return self.root / f"{self.key(backbone, condition)}.features.npy"

    def manifest_path(self, backbone: str, condition: str) -> Path:
        return self.root / f"{self.key(backbone, condition)}.manifest.json"

    def save(
        self,
        backbone: str,
        condition: str,
        features: np.ndarray,
        *,
        provenance: Mapping[str, str],
        extra: Mapping[str, object] | None = None,
    ) -> dict:
        """Write one finished cell. The manifest lands LAST."""
        missing = [k for k in REQUIRED_PROVENANCE if not provenance.get(k)]
        if missing:
            raise ValueError(
                f"falta procedencia {missing}: una celda guardada sin eso no se puede "
                "reusar con seguridad, y la corrida pareceria reanudable sin serlo"
            )
        if condition not in CONDITIONS:
            raise ValueError(f"condicion {condition!r} desconocida")

        features = np.asarray(features, dtype=np.float32)
        manifest = {
            "backbone": backbone,
            "condition": condition,
            "n_rows": int(len(features)),
            "dim": int(features.shape[1]) if features.ndim == 2 else None,
            "features_sha256": _digest(features),
            **{k: provenance[k] for k in REQUIRED_PROVENANCE},
            **(dict(extra) if extra else {}),
        }

        # Features first, manifest last: the manifest is what makes the cell
        # count as done, so it must never arrive before the data it describes.
        _atomic_write(
            self.features_path(backbone, condition), lambda p: np.save(p, features)
        )
        _atomic_write(
            self.manifest_path(backbone, condition),
            lambda p: p.write_text(json.dumps(manifest, indent=1), encoding="utf-8"),
        )
        return manifest

    def load(
        self, backbone: str, condition: str, *, provenance: Mapping[str, str]
    ) -> LoadedCell | None:
        """The cell if it is reusable, else ``None`` -- meaning "run it".

        ``None`` rather than an exception on purpose: unlike the feature cache,
        where an unverifiable entry means something wrote outside the pipeline,
        here every reason to refuse ("older code", "other masks", "other
        split") has the same correct response, which is to compute the cell.
        """
        features_path = self.features_path(backbone, condition)
        manifest_path = self.manifest_path(backbone, condition)
        if not (features_path.is_file() and manifest_path.is_file()):
            return None

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            features = np.load(features_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None

        if manifest.get("backbone") != backbone or manifest.get("condition") != condition:
            return None
        for field in REQUIRED_PROVENANCE:
            if manifest.get(field) != provenance.get(field):
                return None
        if manifest.get("features_sha256") != _digest(features):
            return None

        return LoadedCell(features=features, manifest=manifest)

    def plan(
        self,
        *,
        backbones: Sequence[str],
        conditions: Sequence[str],
        provenance: Mapping[str, str],
        priority_backbones: Sequence[str] = (),
    ) -> dict:
        """Which cells are already done, and the order to run the rest.

        ``priority_backbones`` goes first because the grid answers one
        question before it answers twenty-four: DINOv2 public against the
        proposed backbone is what the professor actually asked, and running it
        first means a crash at cell 20 still leaves that answer on disk.
        """
        unknown = [c for c in conditions if c not in CONDITIONS]
        if unknown:
            raise ValueError(
                f"condicion desconocida {unknown}; validas: {sorted(CONDITIONS)}. "
                "Se rechaza antes de pedir GPU, no a la celda 17"
            )

        done: list[tuple[str, str]] = []
        pending: list[tuple[str, str]] = []
        for backbone in backbones:
            for condition in conditions:
                cell = (backbone, condition)
                if self.load(backbone, condition, provenance=provenance) is not None:
                    done.append(cell)
                else:
                    pending.append(cell)

        priority = list(priority_backbones)
        pending.sort(key=lambda cell: (priority.index(cell[0]) if cell[0] in priority else len(priority)))
        return {"done": done, "pending": pending}
