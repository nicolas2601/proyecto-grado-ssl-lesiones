"""Where a C1 mask came from, checked rather than declared.

Split from :mod:`dermssl.c1_provenance` by size: this half answers "is this the
right mask artifact", the other half answers "is this the right pairing of
features to a probe". Both are admissibility policy, but they fail for
different reasons and are consumed by different callers.

C1 is only self-supervised if its masks are, and only comparable to the
published segmentation if the masks come from the published run. Neither fact
survives as a comment, so both are contracts here.

Depends on nothing else in the C1 package, so it can be imported first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dermssl.c1_mask_export import is_sha256


ALLOWED_MASK_PRODUCERS = ("student_r3",)

# WHICH student_r3. Two checkpoints share the name on the pod:
#   final  d81a492  Dice 0,718 ISIC / 0,785 HAM10000 -- "Configuracion principal"
#   v2     91814a9  Dice 0,716 / 0,783 -- max_area_fraction ablation, explicitly
#                   closed as a default ("se cierra el cambio del tope de area")
# Source: docs/24-analisis-final-resultados-y-escenarios-ada.md, run table.
#
# The sha256 guard proves a file did not change; it cannot prove it is the
# RIGHT file, because hashing the ablation matches itself perfectly. So the run
# identity is declared in the manifest and checked here, as a contract rather
# than a comment someone has to read.
PUBLISHED_SEGMENTATION = ("final", "d81a492")

# The sha256 of salidas/final/student_r3/best.pt. Deliberately None until it is
# MEASURED on the pod and the file is shown to behave like the published run:
#   a) sha256 of the file + its mtime against 2026-09-21-final/final.log
#   b) Dice over ~20 ISIC 2018 images with reference masks, inside the published
#      range (0,718 [0,709; 0,726], docs/24 §4.3). The reference masks are used
#      ONLY for this check and never enter C1.
# Then this constant is set, citing the measured Dice and the date.
#
# While it is None every manifest is REFUSED. Failing closed is the point: the
# previous version let the producer assert its own run identity, so an artifact
# built from the v2 ablation was accepted as "final" simply because the default
# said so. A field the producer fills cannot testify about the producer.
FINAL_STUDENT_R3_SHA256: str | None = None


def checkpoint_registry() -> dict[str, tuple[str, str]]:
    """sha256 -> (run, commit). Empty while the checkpoint is unmeasured."""
    if FINAL_STUDENT_R3_SHA256 is None:
        return {}
    return {FINAL_STUDENT_R3_SHA256: PUBLISHED_SEGMENTATION}


def resolve_run(checkpoint_sha256: str) -> tuple[str, str]:
    """Derive the run from the checkpoint's CONTENT, never from a label."""
    registry = checkpoint_registry()
    if not registry:
        raise ValueError(
            "el checkpoint de la corrida final todavia no esta registrado: "
            "FINAL_STUDENT_R3_SHA256 sigue en None. Medilo en el pod y verificalo "
            "por Dice contra docs/24 antes de correr nada."
        )
    if checkpoint_sha256 not in registry:
        raise ValueError(
            f"checkpoint_sha256 {checkpoint_sha256[:12]}... no corresponde a ninguna "
            "corrida registrada. El checkpoint de la ablacion v2 carga igual de bien "
            "que el publicado; por eso la corrida se deriva del contenido y no de una "
            "etiqueta que el productor se pone a si mismo."
        )
    return registry[checkpoint_sha256]


@dataclass(frozen=True)
class MaskSource:
    """Where a mask came from, checked rather than declared.

    C1 is only self-supervised if the masks are. Rather than trusting a caller
    to pass the right array, a mask has to arrive with the manifest of the
    step-0 artifact, and a manifest from a supervised producer cannot be
    constructed at all. The SSL rule stops being a paragraph in a plan and
    becomes something the code refuses to run without.
    """

    producer: str
    masks_sha256: str
    image_ids: tuple[str, ...]
    mask_shape: tuple[int, int]
    segmentation_run: str = ""
    segmentation_commit: str = ""
    partial: bool = False

    @classmethod
    def from_manifest(cls, manifest: dict, *, allow_partial: bool = False) -> MaskSource:
        producer = manifest.get("producer")
        if producer not in ALLOWED_MASK_PRODUCERS:
            raise ValueError(
                f"mask producer {producer!r} is not allowed; C1 accepts only "
                f"{ALLOWED_MASK_PRODUCERS}. Reference masks would make the classifier "
                "label-dependent and the arm would stop being self-supervised."
            )
        if not manifest.get("producer_is_label_free"):
            raise ValueError(
                f"manifest for {producer!r} does not assert producer_is_label_free"
            )
        checkpoint_sha = manifest.get("checkpoint_sha256")
        if not checkpoint_sha:
            raise ValueError(
                "el manifiesto no trae checkpoint_sha256; sin el no se puede derivar "
                "de que corrida salieron las mascaras"
            )
        if not isinstance(checkpoint_sha, str):
            raise TypeError(
                f"checkpoint_sha256 debe ser str, no {type(checkpoint_sha).__name__}. "
                "Coercionarlo con str() aceptaria cualquier objeto cuyo __str__ "
                "devuelva un digest registrado: la conversion implicita repara el "
                "valor en vez de rechazarlo."
            )
        expected_run, expected_commit = resolve_run(checkpoint_sha)

        run = manifest.get("segmentation_run")
        commit = manifest.get("segmentation_commit")
        if run != expected_run:
            raise ValueError(
                f"segmentation_run dice {run!r} pero el sha del checkpoint corresponde "
                f"a {expected_run!r}: el manifiesto se contradice con el archivo. "
                "The 'v2' checkpoint is the max_area_fraction ablation that lost on "
                "Dice and was closed as a default (docs/24); conditioning C1 on it "
                "would mean the published segmentation quality does not describe the "
                "masks the classifier saw."
            )
        if commit != expected_commit:
            raise ValueError(
                f"segmentation_commit is {commit!r} but run {run!r} was published at "
                f"{expected_commit!r}. The manifest contradicts itself."
            )

        digest = str(manifest["masks_sha256"])
        if not is_sha256(digest):
            raise ValueError(
                f"masks_sha256 is not a lowercase sha256 hex digest: {digest!r}"
            )

        partial = bool(manifest.get("partial", False))
        if partial and not allow_partial:
            raise ValueError(
                "this mask artifact is marked partial (built with --limit or an "
                "unexpected image count). A smoke artifact must not silently become "
                "the input to the 24 cells; pass allow_partial=True to opt in."
            )

        return cls(
            segmentation_run=run,
            segmentation_commit=commit,
            partial=partial,
            producer=producer,
            masks_sha256=digest,
            image_ids=tuple(manifest["image_ids"]),
            mask_shape=tuple(manifest["mask_shape"]),
        )


@dataclass(frozen=True)
class TrackedMask:
    """A boolean mask bound to the artifact and image it belongs to."""

    array: np.ndarray
    source: MaskSource
    image_id: str

    def __post_init__(self) -> None:
        if self.array.dtype != np.bool_:
            raise TypeError(
                f"mask dtype is {self.array.dtype}, expected bool. Convert through "
                "lesion_mask() so 255 is resolved at one boundary; inverting a uint8 "
                "mask with ~ yields 254/255 and indexes instead of selecting."
            )
        if self.image_id not in self.source.image_ids:
            raise ValueError(
                f"image_id {self.image_id!r} is absent from the mask manifest; "
                "the mask and the image may not correspond"
            )
