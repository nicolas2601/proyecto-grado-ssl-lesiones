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

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from dermssl.c1_mask_export import RUN_PRODUCERS, is_sha256


ALLOWED_MASK_PRODUCERS = ("student_r3", "s4_unet_a2")

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

# The inference that produced the published Dice. A checkpoint does not identify
# a mask: the published path normalises with ImageNet statistics and
# ``dermssl.torch_unet.infer_mask`` does not, TTA averages four flipped views,
# and the threshold is a free parameter. Same weights, different masks, no error.
#
# These are registered values rather than imports on purpose. This module stays
# torch-free and importable from ``src`` alone, while ``evaluate_segmentation``
# lives under plan-segmentacion and needs two more entries on sys.path -- adding
# them here would make the C1 package unimportable anywhere the segmentation
# tree is absent. The link to the source is a test instead
# (TestThePublishedInferenceMatchesThePublishedModules), the same way the
# checkpoint sha is measured rather than derived.
PUBLISHED_INFERENCE: Mapping[str, Any] = MappingProxyType(
    {
        "path": "plan-segmentacion/comun/evaluacion/eval_student.predict_masks",
        "tta": False,
        # config["resolution"] of the final checkpoint, read on the pod 2026-09-26.
        "resolution": 224,
        "threshold": 0.5,
        # Probabilities resized to the original grid, THEN thresholded. The other
        # order interpolates between two labels, which is a second, undeclared
        # threshold: an isolated 0,55 pixel covers 1 pixel one way and 5 the other.
        "upsample": "bilinear_prob_then_threshold",
    }
)

# The sha256 of salidas/final/student_r3/best.pt, MEASURED on the pod on
# 2026-09-26 and verified to behave like the published run:
#
#   path   /home/jovyan/work/proyectonp/salidas/final/student_r3/best.pt
#   mtime  2026-09-21 09:08:04 UTC, 31.132.335 bytes -- the timestamp the final
#          run wrote it, which final.log references
#   check  evaluate_segmentation.py (published, unmodified, no --tta) over the
#          first 20 ids of salidas/final/evaluacion_isic2018/per_image.csv:
#          20/20 matched, mean |delta| 3,2e-06, max |delta| 4,08e-05 against the
#          PUBLISHED per-image Dice. The residue is cuDNN non-determinism.
#   scope  docs/24 §4.3 reports 0,718 [0,709; 0,726] over the whole set; the
#          same CSV averages 0,7177 over its 2594 rows, so the published number
#          is verified too.
#
# The check is per-image and PAIRED, not a comparison of means: those 20 images
# average 0,855 in the published CSV, so "within 0,05 of 0,718" would have
# failed a perfect reproduction. With a per-image std near 0,25 the standard
# error at n=20 is ~0,056, and a mean-vs-mean test fails roughly one run in
# three even when nothing is wrong. Same id, same checkpoint, same script: the
# number has to come out identical, not close.
#
# The reference masks are used ONLY for this check and never enter C1.
#
# While it is None every manifest is REFUSED. Failing closed was the point: an
# earlier version let the producer assert its own run identity, so an artifact
# built from the v2 ablation was accepted as "final" simply because the default
# said so. A field the producer fills cannot testify about the producer.
FINAL_STUDENT_R3_SHA256: str | None = (
    "ac87b133dd50e74a88d9137b2f1ba76b986ee176017b8b009b0caf52a00c5cc0"
)


# C1-v7 SECONDARY (docs/27): S4-UNet A2 met its prediction, so the C1 grid is
# re-run once with its masks. Same fail-closed rule as student_r3: while the
# sha256 of salidas/s4_unet/a2/best.pt and the code revision that trained it
# (run.json "code_revision") are unmeasured, an A2 artifact is REFUSED. Both
# are measured on the pod and committed here; see scripts/c1_predict_masks.py.
S4_UNET_A2_RUN = "s4_unet_a2"
S4_UNET_A2_SHA256: str | None = "95bec0dc02814bdb68f37b617558eb4ca3a2920680e2dde69d3e0a81eae0260f"
S4_UNET_A2_COMMIT: str | None = "842eddf17b57b6c62d12d17496043596a3e1d3e2"


def checkpoint_registry() -> dict[str, tuple[str, str]]:
    """sha256 -> (run, commit), only for the checkpoints already measured."""
    registry: dict[str, tuple[str, str]] = {}
    if FINAL_STUDENT_R3_SHA256 is not None:
        registry[FINAL_STUDENT_R3_SHA256] = PUBLISHED_SEGMENTATION
    if S4_UNET_A2_SHA256 is not None and S4_UNET_A2_COMMIT is not None:
        registry[S4_UNET_A2_SHA256] = (S4_UNET_A2_RUN, S4_UNET_A2_COMMIT)
    return registry


def producer_for_run(run: str) -> str:
    """The producer a run's masks are recorded under (``c1_mask_export``)."""
    return RUN_PRODUCERS[run]


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


def check_inference(declared: object) -> dict[str, Any]:
    """Refuse anything that is not the published inference, field by field.

    Equality against the whole block would say "no coincide" and leave the
    reader to diff two dicts; naming the field is what makes the failure
    actionable, and ``tta`` in particular has to be compared by identity
    because ``"false" == False`` is False but ``bool("false")`` is True.
    """
    if not isinstance(declared, Mapping):
        raise ValueError(
            "el manifiesto no trae el bloque 'inference'. Un checkpoint no "
            "identifica una mascara: el camino publicado normaliza con "
            "estadisticas de ImageNet y dermssl.torch_unet.infer_mask no, TTA "
            "promedia cuatro vistas y el umbral es un parametro libre. Sin ese "
            "bloque el manifiesto describe un archivo, no un resultado."
        )

    missing = [key for key in PUBLISHED_INFERENCE if key not in declared]
    if missing:
        raise ValueError(f"al bloque 'inference' le faltan campos: {missing}")

    for key, expected in PUBLISHED_INFERENCE.items():
        found = declared[key]
        if key == "tta":
            if found is not False:
                raise ValueError(
                    f"inference['tta'] es {found!r}; C1 corre sin TTA porque el "
                    "Dice publicado es el numero sin TTA. Con TTA las mascaras "
                    "son mejores que las que reporta la tesis."
                )
            continue
        if type(found) is not type(expected) or found != expected:
            raise ValueError(
                f"inference[{key!r}] es {found!r} y la corrida publicada usa "
                f"{expected!r}. Cambiarlo produce mascaras distintas con el mismo "
                "checkpoint, sin ningun error."
            )
    return dict(PUBLISHED_INFERENCE)


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
    # Pairs rather than a dict so the frozen dataclass stays hashable; read it
    # through the ``inference`` property.
    inference_items: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    @property
    def inference(self) -> dict[str, Any]:
        return dict(self.inference_items)

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
        if producer != producer_for_run(expected_run):
            raise ValueError(
                f"producer dice {producer!r} pero el sha del checkpoint corresponde a "
                f"{producer_for_run(expected_run)!r}: las mascaras de un segmentador no "
                "pueden entrar a C1 con el nombre del otro."
            )

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

        # Round 6 fixed this for checkpoint_sha256 and left the field one line
        # over unguarded: str() on an arbitrary object is a coercion, not a
        # check, so anything whose __str__ prints a digest becomes that digest.
        # This hash is what ties a published number to a file on disk.
        digest = manifest["masks_sha256"]
        if not isinstance(digest, str):
            raise TypeError(
                f"masks_sha256 debe ser str, no {type(digest).__name__}. "
                "Coercionarlo con str() aceptaria cualquier objeto cuyo __str__ "
                "devuelva algo con forma de digest: la conversion implicita repara "
                "el valor en vez de rechazarlo."
            )
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

        inference = check_inference(manifest.get("inference"))

        return cls(
            segmentation_run=run,
            segmentation_commit=commit,
            partial=partial,
            inference_items=tuple(sorted(inference.items())),
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
