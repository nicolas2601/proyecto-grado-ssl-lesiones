"""Contratos JSON para ejecuciones SSL reproducibles y comparables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dermssl.pretrain import config as config_mod

SCHEMA_VERSION = "1.0"
STAGES = ("base", "adapted")


@dataclass(frozen=True)
class RunVariant:
    """Un estado de pesos en una comparación SSL pareada por backbone."""

    name: str
    backbone: str
    stage: str
    split_hash: str
    seed: int
    probe: str
    weights: str


@dataclass(frozen=True)
class CheckpointEvaluation:
    """Evidencia solo de validación para elegir un checkpoint adaptado."""

    checkpoint: str
    split_hash: str
    seed: int
    probe: str
    evaluation_split: str
    macro_f1: float
    global_auc: float
    tone_gap: float


def build_run_manifest(
    cfg: config_mod.PretrainConfig,
    *,
    gpu_count: int,
    data_revision: str = "",
    code_revision: str = "",
    transformed_manifest: config_mod.TransformedManifestEvidence | None = None,
) -> dict:
    """Devuelve el registro de procedencia JSON principal antes de entrenar."""
    cfg = config_mod.validate(cfg)
    if gpu_count < 0:
        raise ValueError(f"gpu_count no puede ser negativo; se recibió {gpu_count}")
    if gpu_count == 0 and cfg.max_steps == 0:
        raise ValueError(
            "las ejecuciones sin GPU requieren max_steps positivo para evitar "
            "un entrenamiento CPU sin límite"
        )
    if cfg.adaptation_mode == "original_transformed_consistency" and transformed_manifest is None:
        raise ValueError("la ejecución pareada requiere evidencia del manifiesto transformado")
    if transformed_manifest is not None and cfg.adaptation_mode != "original_transformed_consistency":
        raise ValueError("la evidencia del manifiesto transformado requiere adaptación pareada")
    return {
        "schema_version": SCHEMA_VERSION,
        "split_hash": cfg.split_hash,
        "data_revision": data_revision or cfg.data_revision or None,
        "code_revision": code_revision or None,
        "backbone": cfg.backbone,
        "method": cfg.method,
        "initial_weights": cfg.initial_weights,
        "resume_checkpoint": cfg.resume_checkpoint or None,
        "adaptation_mode": cfg.adaptation_mode,
        "transformed_image_root": cfg.transformed_image_root or None,
        "transformed_data_revision": cfg.transformed_data_revision or None,
        "transformed_manifest_path": (
            transformed_manifest.manifest_path if transformed_manifest is not None else None
        ),
        "transformed_manifest_revision": (
            transformed_manifest.manifest_revision if transformed_manifest is not None else None
        ),
        "transformed_manifest_sha256": (
            transformed_manifest.manifest_sha256 if transformed_manifest is not None else None
        ),
        "consistency_weight": cfg.consistency_weight,
        "max_steps": cfg.max_steps or None,
        "seed": cfg.seed,
        "gpu_count": gpu_count,
        "precision": cfg.precision,
        "budget_gpu_hours": cfg.budget_gpu_hours,
        "checkpoint_fractions": list(cfg.checkpoint_fractions),
        "label_free": True,
    }


def write_run_manifest(manifest: dict, path: str | Path) -> None:
    """Escribe JSON determinista para que cada checkpoint tenga un contrato legible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def record_execution_evidence(
    manifest: dict,
    *,
    device: str,
    steps_executed: int,
    image_count: int,
    pair_count: int,
    ssl_loss: float | None,
    consistency_loss: float | None,
    dataset_image_count: int | None = None,
    dataset_pair_count: int | None = None,
    samples_processed: int | None = None,
    total_loss: float | None = None,
) -> dict:
    """Añade al contrato la evidencia observada, sin sustituir su procedencia.

    El manifiesto inicial se escribe antes de reservar el dispositivo; esta
    actualización final conserva las claves existentes y registra lo que la
    ejecución realmente realizó, incluso cuando el smoke cae a CPU.

    ``image_count`` se conserva como cardinalidad heredada del dataset.
    ``dataset_image_count`` y ``dataset_pair_count`` declaran explícitamente
    las cardinalidades; ``samples_processed`` y ``pair_count`` acreditan lo
    realmente procesado durante esta ejecución.
    """
    if not device:
        raise ValueError("device debe identificar el dispositivo realmente usado")
    dataset_image_count = image_count if dataset_image_count is None else dataset_image_count
    dataset_pair_count = pair_count if dataset_pair_count is None else dataset_pair_count
    samples_processed = image_count if samples_processed is None else samples_processed
    if min(
        steps_executed,
        image_count,
        pair_count,
        dataset_image_count,
        dataset_pair_count,
        samples_processed,
    ) < 0:
        raise ValueError("los conteos de ejecución no pueden ser negativos")
    if manifest.get("adaptation_mode") == "original_transformed_consistency":
        if not manifest.get("transformed_image_root"):
            raise ValueError("la ejecución pareada requiere transformed_image_root")
        if not manifest.get("transformed_manifest_sha256"):
            raise ValueError("la ejecución pareada debe vincular el hash del manifiesto")
        if dataset_pair_count != dataset_image_count:
            raise ValueError("el conjunto pareado debe registrar un par por imagen")
        if pair_count != samples_processed:
            raise ValueError("pair_count debe registrar los pares realmente procesados")
        if consistency_loss is None:
            raise ValueError("la ejecución pareada debe registrar consistency_loss")
        if ssl_loss is None or total_loss is None:
            raise ValueError("la ejecución pareada debe separar ssl_loss y total_loss")
    return {
        **manifest,
        "execution": {
            "device": device,
            "steps_executed": steps_executed,
            "image_count": image_count,
            "dataset_image_count": dataset_image_count,
            "dataset_pair_count": dataset_pair_count,
            "samples_processed": samples_processed,
            "pair_count": pair_count,
            "ssl_loss": ssl_loss,
            "consistency_loss": consistency_loss,
            "total_loss": total_loss,
        },
    }


def validate_paired_variants(
    variants: Iterable[RunVariant],
) -> tuple[RunVariant, ...]:
    """Rechaza matrices que cambian split, seed, probe o etapas."""
    rows = tuple(variants)
    if not rows:
        raise ValueError("la comparación pareada requiere al menos una variante")
    split_hashes = {row.split_hash for row in rows}
    seeds = {row.seed for row in rows}
    probes = {row.probe for row in rows}
    if len(split_hashes) != 1:
        raise ValueError("las variantes pareadas deben compartir un split_hash")
    if len(seeds) != 1:
        raise ValueError("las variantes pareadas deben compartir una seed")
    if len(probes) != 1:
        raise ValueError("las variantes pareadas deben compartir un probe")

    by_backbone: dict[str, set[str]] = {}
    for row in rows:
        if row.stage not in STAGES:
            raise ValueError(f"etapa de comparación desconocida {row.stage!r}")
        if not row.weights:
            raise ValueError(f"la variante {row.name!r} debe indicar sus pesos")
        by_backbone.setdefault(row.backbone, set()).add(row.stage)
    incomplete = sorted(
        backbone for backbone, stages in by_backbone.items() if set(STAGES) - stages
    )
    if incomplete:
        raise ValueError(f"backbones sin variantes base/adapted: {incomplete}")
    return rows


def select_validation_checkpoint(
    baseline: CheckpointEvaluation,
    candidates: Iterable[CheckpointEvaluation],
    *,
    max_auc_drop: float = 0.02,
) -> CheckpointEvaluation:
    """Selecciona el mejor checkpoint de validación que supera las puertas de equidad."""
    rows = tuple(candidates)
    _validate_validation_evaluation(baseline)
    eligible = []
    for candidate in rows:
        _validate_validation_evaluation(candidate)
        if candidate.split_hash != baseline.split_hash:
            raise ValueError("las evaluaciones de checkpoint deben compartir un split_hash")
        if candidate.seed != baseline.seed:
            raise ValueError("las evaluaciones de checkpoint deben compartir una seed")
        if candidate.probe != baseline.probe:
            raise ValueError("las evaluaciones de checkpoint deben compartir un probe")
        auc_drop = baseline.global_auc - candidate.global_auc
        if (
            candidate.macro_f1 > baseline.macro_f1
            and auc_drop <= max_auc_drop
            and candidate.tone_gap < baseline.tone_gap
        ):
            eligible.append(candidate)
    if not eligible:
        raise ValueError("ningún checkpoint mejora el macro-F1 de validación dentro de las puertas de equidad")
    return min(eligible, key=lambda item: (-item.macro_f1, item.checkpoint))


def build_checkpoint_promotion_record(
    baseline: CheckpointEvaluation,
    candidates: Iterable[CheckpointEvaluation],
    *,
    max_auc_drop: float = 0.02,
) -> dict[str, str | float]:
    """Construye un registro determinista y solo de validación para promover checkpoints."""
    selected = select_validation_checkpoint(
        baseline, candidates, max_auc_drop=max_auc_drop
    )
    return {
        "selection_split": "validation",
        "selected_checkpoint": selected.checkpoint,
        "max_auc_drop": max_auc_drop,
        "selection_rule": "max_validation_macro_f1_with_auc_and_tone_gap_gates",
    }


def _validate_validation_evaluation(evaluation: CheckpointEvaluation) -> None:
    if evaluation.evaluation_split != "validation":
        raise ValueError("la selección de checkpoint debe usar solo evidencia de validación")
