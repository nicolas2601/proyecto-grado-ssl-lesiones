"""Configuración y puertas estrictas para las ejecuciones SSL v6.

Motivación
----------
v5 demostró que la lógica de aceptación no debe vivir en un notebook ni en un
script de lanzamiento: el código sin pruebas incorpora errores. v6 entrena en
un clúster multi-GPU, donde una ejecución desperdiciada consume horas-GPU; por
eso cada decisión validable de forma económica se valida aquí, antes de
importar torch.

El invariante central es que el preentrenamiento no use etiquetas. Este módulo
lo hace ejecutable: la configuración no expresa archivos de etiquetas,
``assert_label_free`` inspecciona los campos como defensa adicional y
``collect_images`` devuelve únicamente archivos de imagen.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from dermssl import patch_guard
from dermssl.pretrain.levers import Levers

METHODS = (
    "simsiam",
    "simsiam_imagenet",
    "simsiam_dinov2",
    "dino_continual",
    "dino_scratch",
)
# dino_scratch: mismo ViT y objetivo que dino_continual, con inicio aleatorio.
# Es el brazo con una sola variable aislada (los pesos iniciales); SimSiam se
# mantiene como ablación secundaria (docs/16, sección 2).
# simsiam_dinov2: método propuesto para la tesis, con objetivo SimSiam sobre
# DINOv2 ViT-S/14 desde los pesos públicos sin etiquetas.
DINO_METHODS = ("dino_continual", "dino_scratch")
SIMSIAM_METHODS = ("simsiam", "simsiam_imagenet", "simsiam_dinov2")
DEFAULT_BACKBONES = {
    "simsiam": "resnet50",
    "simsiam_imagenet": "resnet50",
    "simsiam_dinov2": "dinov2_vits14",
    "dino_continual": "dinov2_vits14",
    "dino_scratch": "dinov2_vits14",
}
DEFAULT_INITIAL_WEIGHTS = {
    "simsiam": "random",
    "simsiam_imagenet": "imagenet1k_v2",
    "simsiam_dinov2": "dinov2-public-lvd142m",
    "dino_continual": "dinov2-public-lvd142m",
    "dino_scratch": "random",
}
PRECISIONS = ("bf16", "fp16", "fp32")
ADAPTATION_MODES = ("two_view_ssl", "original_transformed_consistency")
GRAD_CLIP_NORM = 3.0  # Max grad norm, recorded in run.json with the rest.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Las tasas base siguen la regla de escalado lineal de cada artículo: SimSiam
# usa 0.05 x batch/256 (SGD desde cero). El preentrenamiento continuo inicia
# con pesos DINOv2 convergidos, por lo que su LR base es 5e-4 x batch/1024
# (AdamW). Ambas son provisionales hasta contar con una corrida DINOv2 real.
LR_RULES = {
    "simsiam": (0.05, 256),
    "simsiam_imagenet": (5e-4, 1024),
    "simsiam_dinov2": (5e-4, 1024),  # Regla de pesos convergidos, como continual.
    "dino_continual": (5e-4, 1024),
    "dino_scratch": (5e-4, 256),  # Regla del artículo DINO desde cero.
}

# Cualquier campo que parezca supervisión activa la puerta.
FORBIDDEN_FIELD_TOKENS = ("label", "target", "class", "dx")
FORBIDDEN_VALUE_SUFFIXES = (".csv", ".tsv", ".parquet")


class LabelLeakError(RuntimeError):
    """Se lanza cuando la configuración de preentrenamiento recibe etiquetas."""


@dataclass(frozen=True)
class TransformedManifestEvidence:
    """Pares y procedencia verificados para una adaptación original/transformada."""

    pairs: tuple[tuple[Path, Path], ...]
    manifest_path: str
    manifest_revision: str
    manifest_sha256: str


@dataclass(frozen=True)
class PretrainConfig:
    method: str
    image_root: str
    output_dir: str
    split_hash: str = ""
    backbone: str = ""  # Vacío: validate() lo resuelve desde method.
    initial_weights: str = ""  # Vacío: validate() lo resuelve desde method.
    resume_checkpoint: str = ""
    data_revision: str = ""
    transformed_image_root: str = ""
    transformed_manifest_path: str = ""
    transformed_data_revision: str = ""
    adaptation_mode: str = "two_view_ssl"
    consistency_weight: float = 1.0
    max_steps: int = 0
    resolution: int = 224
    local_resolution: int = 98  # 7 x 14; el valor DINO 96 rompe el parche 14.
    n_local_crops: int = 6
    batch_size_per_gpu: int = 64
    budget_gpu_hours: float = 8.0
    checkpoint_fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
    warmup_fraction: float = 0.05
    weight_decay: float = 1e-4
    momentum_teacher: float = 0.996  # Solo dino_continual (valor inicial EMA).
    seed: int = 42
    num_workers: int = 8
    precision: str = "bf16"
    color_constancy_probs: tuple[tuple[str, float], ...] = (
        ("identity", 0.4),
        ("gray_world", 0.3),
        ("shades_of_gray", 0.3),
    )
    lesion_guided_crops: bool = True
    # docs/28 levers; every default reproduces the v6 run exactly.
    freeze_blocks: int = 0
    layer_decay: float = 1.0
    anchor_l2sp: float = 0.0
    anchor_distill: float = 0.0
    tone_consistency_weight: float = 0.0
    hue_jitter: float = 0.1
    exclude_ids_file: str = ""  # Ids (one per line or JSON list) kept out of the pool.
    mlflow_experiment: str = "obj2-ssl-pretraining"
    tracking_uri: str = ""  # Vacío: valor predeterminado de MLflow (archivo o entorno).


def validate(cfg: PretrainConfig) -> PretrainConfig:
    """Devuelve una copia normalizada de ``cfg`` o un mensaje accionable."""
    if cfg.method not in METHODS:
        raise ValueError(f"método desconocido {cfg.method!r}; se esperaba uno de {METHODS}")
    if cfg.precision not in PRECISIONS:
        raise ValueError(f"precisión desconocida {cfg.precision!r}; se esperaba una de {PRECISIONS}")
    if cfg.adaptation_mode not in ADAPTATION_MODES:
        raise ValueError(
            f"adaptation_mode desconocido {cfg.adaptation_mode!r}; "
            f"se esperaba uno de {ADAPTATION_MODES}"
        )
    if (
        cfg.adaptation_mode == "original_transformed_consistency"
        and not cfg.transformed_image_root
    ):
        raise ValueError(
            "transformed_image_root es obligatorio para la adaptación original/transformada"
        )
    if (
        cfg.adaptation_mode == "original_transformed_consistency"
        and cfg.method not in SIMSIAM_METHODS
    ):
        raise ValueError(
            "la adaptación original/transformada solo es compatible con métodos SimSiam"
        )
    if (
        cfg.adaptation_mode == "original_transformed_consistency"
        and not cfg.transformed_manifest_path
    ):
        raise ValueError(
            "transformed_manifest_path es obligatorio para la adaptación original/transformada"
        )
    if (
        cfg.adaptation_mode == "original_transformed_consistency"
        and not cfg.transformed_data_revision
    ):
        raise ValueError(
            "transformed_data_revision debe declarar la revisión del manifiesto transformado"
        )
    if (
        cfg.adaptation_mode == "original_transformed_consistency"
        and not cfg.data_revision
    ):
        raise ValueError(
            "data_revision debe declarar la revisión fuente para la adaptación original/transformada"
        )
    if cfg.consistency_weight <= 0:
        raise ValueError("consistency_weight debe ser positivo")
    _validate_levers(cfg)
    if cfg.max_steps < 0:
        raise ValueError("max_steps debe ser cero o positivo")
    if cfg.budget_gpu_hours <= 0:
        raise ValueError(f"budget_gpu_hours debe ser positivo; se recibió {cfg.budget_gpu_hours}")
    if len(cfg.split_hash) != 64 or any(
        char not in "0123456789abcdef" for char in cfg.split_hash.lower()
    ):
        raise ValueError("split_hash debe ser el SHA-256 de 64 caracteres de la partición congelada")
    _validate_fractions(cfg.checkpoint_fractions)

    backbone = cfg.backbone or DEFAULT_BACKBONES[cfg.method]
    initial_weights = cfg.initial_weights or DEFAULT_INITIAL_WEIGHTS[cfg.method]
    patch_guard.assert_compatible(cfg.resolution, backbone)
    if cfg.method in DINO_METHODS:
        patch_guard.assert_compatible(cfg.local_resolution, backbone)

    normalized = dataclasses.replace(
        cfg, backbone=backbone, initial_weights=initial_weights
    )
    assert_label_free(normalized)
    return normalized


def levers_of(cfg: PretrainConfig):
    return Levers(
        freeze_blocks=cfg.freeze_blocks,
        layer_decay=cfg.layer_decay,
        anchor_l2sp=cfg.anchor_l2sp,
        anchor_distill=cfg.anchor_distill,
        tone_consistency_weight=cfg.tone_consistency_weight,
    )


def _validate_levers(cfg: PretrainConfig) -> None:
    levers = levers_of(cfg).validate(
        paired=cfg.adaptation_mode == "original_transformed_consistency"
    )
    if not levers.is_default() and cfg.method != "simsiam_dinov2":
        raise ValueError(
            f"las palancas de docs/28 solo aplican a simsiam_dinov2 (ViT); método {cfg.method!r}"
        )
    if not 0.0 <= cfg.hue_jitter <= 0.5:
        raise ValueError(f"hue_jitter={cfg.hue_jitter}: ColorJitter exige [0, 0.5]")
    if cfg.hue_jitter != PretrainConfig.hue_jitter and cfg.method not in SIMSIAM_METHODS:
        raise ValueError(
            f"hue_jitter solo llega al dataset SimSiam; en {cfg.method!r} se registraría sin efecto"
        )


def load_excluded_ids(path: str | Path) -> frozenset[str]:
    """Ids to keep out of the pool: a JSON list or one id per line."""
    text = Path(path).read_text(encoding="utf-8").strip()
    ids = json.loads(text) if text.startswith("[") else text.splitlines()
    ids = frozenset(i.strip() for i in ids if i.strip())
    if not ids:
        raise ValueError(f"{path}: no contiene ids; excluir nada en silencio no es una exclusión")
    return ids


def _original(entry) -> Path:
    return Path(entry[0] if isinstance(entry, tuple) else entry)


def _content_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_exclusion(paths: list, ids: frozenset[str]) -> tuple[list, dict]:
    """Drop the excluded ids AND byte-identical copies of them under other ids.

    Review P1-1: matching by stem alone excluded nothing, silently, when the
    names did not line up, and kept any copy of a validation image that a
    second source ships under another id. Every requested id must be found;
    content copies are caught by size first, so only same-size files are
    hashed.
    """
    found = {_original(e).stem for e in paths} & ids
    missing = sorted(ids - found)
    if missing:
        raise ValueError(
            f"{len(missing)} ids a excluir no aparecen en el pool (p. ej. {missing[:5]}): "
            "una exclusion que no encuentra sus ids no excluye nada"
        )
    by_id = [e for e in paths if _original(e).stem in ids]
    rest = [e for e in paths if _original(e).stem not in ids]
    sizes = {_original(e).stat().st_size for e in by_id}
    digests = {_content_digest(_original(e)) for e in by_id}
    copies = {
        id(e) for e in rest
        if _original(e).stat().st_size in sizes and _content_digest(_original(e)) in digests
    }
    kept = [e for e in rest if id(e) not in copies]
    report = {"n_ids": len(ids), "n_removed_by_id": len(by_id), "n_removed_by_content": len(copies)}
    return kept, report


def _validate_fractions(fractions: tuple[float, ...]) -> None:
    if not fractions or list(fractions) != sorted(fractions):
        raise ValueError(f"las fracciones de checkpoint deben ser ascendentes; se recibió {fractions}")
    if any(f <= 0 or f > 1 for f in fractions):
        raise ValueError(f"las fracciones de checkpoint deben pertenecer a (0, 1]; se recibió {fractions}")
    if fractions[-1] != 1.0:
        raise ValueError(f"la última fracción de checkpoint debe ser 1.0; se recibió {fractions[-1]}")


def assert_label_free(cfg: PretrainConfig) -> None:
    """Defense in depth: no field may carry anything label-shaped.

    The dataclass already has no label field; this guards against a future
    edit adding one, or a path pointing at a metadata table instead of an
    image directory.
    """
    for field in dataclasses.fields(cfg):
        if any(token in field.name.lower() for token in FORBIDDEN_FIELD_TOKENS):
            raise LabelLeakError(
                f"el campo de configuración {field.name!r} parece supervisión; "
                "el preentrenamiento debe permanecer sin etiquetas"
            )
        value = getattr(cfg, field.name)
        if isinstance(value, str) and value.lower().endswith(FORBIDDEN_VALUE_SUFFIXES):
            raise LabelLeakError(
                f"el campo de configuración {field.name!r} apunta a una tabla ({value!r}); "
                "el preentrenamiento solo consume carpetas de imágenes"
            )


def collect_images(root: str | Path) -> list[Path]:
    """Every image under ``root``, and nothing that is not an image.

    Metadata tables (csv/tsv) are ignored by construction, which is the
    structural half of the label-free guarantee: the data pipeline cannot read
    labels because it never sees the files that contain them.

    ``recurse_symlinks=True`` because a pool staged from several read-only
    dataset mounts is, by construction, a directory of symlinks - without it
    ``rglob`` silently returns zero images from every linked dataset instead
    of raising, since only the root itself being a symlink is followed by
    default.
    """
    root = Path(root)
    found = sorted(
        p for p in root.rglob("*", recurse_symlinks=True)
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not found:
        raise FileNotFoundError(f"no se encontraron imágenes ({sorted(IMAGE_SUFFIXES)}) bajo {root}")
    return found


def validate_transformed_manifest(cfg: PretrainConfig) -> TransformedManifestEvidence:
    """Verifica manifiesto, revisiones, conjunto de rutas y hashes de cada par.

    La revisión del manifiesto acredita el contenido canónico que generó CIRCLe,
    mientras que ``manifest_sha256`` fija el archivo exacto consumido por la
    ejecución. Ningún par se entrega al cargador antes de pasar ambas puertas.
    """
    cfg = validate(cfg)
    if cfg.adaptation_mode != "original_transformed_consistency":
        raise ValueError("el manifiesto transformado solo aplica a la adaptación pareada")
    return _validate_transformed_paths(
        original_root=Path(cfg.image_root),
        transformed_root=Path(cfg.transformed_image_root),
        manifest_path=Path(cfg.transformed_manifest_path),
        source_revision=cfg.data_revision,
        transformed_revision=cfg.transformed_data_revision,
    )


def collect_paired_images(
    original_root: str | Path,
    transformed_root: str | Path,
    *,
    manifest_path: str | Path,
    source_revision: str,
    transformed_revision: str,
) -> list[tuple[Path, Path]]:
    """Devuelve únicamente pares cuyo manifiesto y contenido ya fueron validados."""
    evidence = _validate_transformed_paths(
        original_root=Path(original_root),
        transformed_root=Path(transformed_root),
        manifest_path=Path(manifest_path),
        source_revision=source_revision,
        transformed_revision=transformed_revision,
    )
    return list(evidence.pairs)


def _validate_transformed_paths(
    *,
    original_root: Path,
    transformed_root: Path,
    manifest_path: Path,
    source_revision: str,
    transformed_revision: str,
) -> TransformedManifestEvidence:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no existe el manifiesto transformado: {manifest_path}")
    raw_manifest = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
    try:
        payload = json.loads(raw_manifest)
    except json.JSONDecodeError as error:
        raise ValueError(f"el manifiesto transformado no es JSON válido: {manifest_path}") from error
    if not isinstance(payload, dict):
        raise ValueError("el manifiesto transformado debe ser un objeto JSON")

    revision = payload.get("revision")
    canonical_payload = {key: value for key, value in payload.items() if key != "revision"}
    expected_revision = hashlib.sha256(
        json.dumps(canonical_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if revision != expected_revision:
        raise ValueError("la revisión del manifiesto transformado no coincide con su contenido")
    if revision != transformed_revision:
        raise ValueError("la revisión del manifiesto transformado no coincide con transformed_data_revision")
    if payload.get("source_revision") != source_revision:
        raise ValueError("la revisión fuente del manifiesto no coincide con data_revision")

    rows = payload.get("transforms")
    if not isinstance(rows, list):
        raise ValueError("el manifiesto transformado debe contener una lista transforms")
    records: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("cada transformación del manifiesto debe ser un objeto")
        relative_path = row.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("cada transformación debe declarar relative_path")
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"relative_path no es seguro: {relative_path!r}")
        normalized = relative.as_posix()
        if normalized in records:
            raise ValueError(f"relative_path duplicado en el manifiesto transformado: {normalized}")
        records[normalized] = row

    originals = {
        image.relative_to(original_root).as_posix(): image
        for image in collect_images(original_root)
    }
    transformed = {
        image.relative_to(transformed_root).as_posix(): image
        for image in collect_images(transformed_root)
    }
    manifest_paths = set(records)
    original_paths = set(originals)
    transformed_paths = set(transformed)
    if original_paths != manifest_paths or transformed_paths != manifest_paths:
        raise ValueError(
            "las rutas original, transformada y del manifiesto deben coincidir exactamente"
        )

    pairs = []
    for relative_path in sorted(manifest_paths):
        row = records[relative_path]
        original = originals[relative_path]
        transformed_image = transformed[relative_path]
        if row.get("source_sha256") != _sha256_file(original):
            raise ValueError(f"el hash original no coincide para {relative_path}")
        if row.get("transformed_sha256") != _sha256_file(transformed_image):
            raise ValueError(f"el hash transformado no coincide para {relative_path}")
        pairs.append((original, transformed_image))

    return TransformedManifestEvidence(
        pairs=tuple(pairs),
        manifest_path=str(manifest_path.resolve()),
        manifest_revision=revision,
        manifest_sha256=manifest_sha256,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def effective_lr(cfg: PretrainConfig, world_size: int) -> float:
    """Linear-scaling learning rate for the total batch across all GPUs."""
    base, denominator = LR_RULES[cfg.method]
    total_batch = cfg.batch_size_per_gpu * world_size
    return base * total_batch / denominator


def mlflow_params(cfg: PretrainConfig) -> dict[str, str | int | float | bool]:
    """Flatten the config into MLflow-safe primitives.

    Tuples become comma-joined strings so every run in the UI shows the exact
    values without truncation or repr noise.
    """
    params: dict[str, str | int | float | bool] = {}
    for field in dataclasses.fields(cfg):
        value = getattr(cfg, field.name)
        if isinstance(value, tuple):
            if value and isinstance(value[0], tuple):
                value = ",".join(f"{name}:{prob}" for name, prob in value)
            else:
                value = ",".join(str(v) for v in value)
        params[field.name] = value
    return params
