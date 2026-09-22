"""Write the release notes from the run manifests, not from memory.

The previous notes said "DINOv2 ViT-S/14" in prose. That was true of the
first run and became false the moment a ViT-B arm was added - the kind of
claim nobody re-reads once it is written. Here every fact about a published
checkpoint is read from its own ``run.json``, so the notes cannot describe a
model the release does not contain.

The other rule this file enforces: a checkpoint without its ``run.json`` is
not publishable. A .pt on its own cannot be reproduced or cited - nobody
knows which data, which budget or which split produced it.

    python scripts/build_release_notes.py --tag v1.0-tone --dir <staging>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WEIGHTS_SUFFIX = "_backbone.pt"
RUN_SUFFIX = "_run.json"
NESTED_WEIGHTS = "simsiam_dinov2_backbone_final.pt"
NESTED_RUN = "run.json"


class MissingEvidence(Exception):
    """A checkpoint arrived without the manifest that makes it citable."""


def discover_arms(directory: Path) -> dict[str, dict[str, Path]]:
    """Map each arm to its weights and manifest, flat layout or nested.

    Flat is what a downloaded artifact looks like (``<arm>_backbone.pt``);
    nested is the training output on the pod (``<arm>/run.json``). Both are
    accepted so the same script works from either side.
    """
    found: dict[str, dict[str, Path]] = {}

    for weights in sorted(directory.glob(f"*{WEIGHTS_SUFFIX}")):
        arm = weights.name[: -len(WEIGHTS_SUFFIX)]
        manifest = directory / f"{arm}{RUN_SUFFIX}"
        if not manifest.is_file():
            raise MissingEvidence(f"{weights.name} llego sin {manifest.name}")
        found[arm] = {"weights": weights, "run": manifest}

    for sub in sorted(p for p in directory.iterdir() if p.is_dir()):
        weights = sub / NESTED_WEIGHTS
        if not weights.is_file():
            continue
        manifest = sub / NESTED_RUN
        if not manifest.is_file():
            raise MissingEvidence(f"{sub.name}/{NESTED_WEIGHTS} llego sin {NESTED_RUN}")
        found.setdefault(sub.name, {"weights": weights, "run": manifest})

    if not found:
        raise MissingEvidence(f"no hay ningun checkpoint publicable en {directory}")
    return found


def _read(manifest: Path) -> dict:
    return json.loads(manifest.read_text(encoding="utf-8"))


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "si" if value else "no"
    if isinstance(value, (int,)):
        return f"{value:,}".replace(",", ".")
    if isinstance(value, float):
        return f"{value:.{digits}f}".replace(".", ",")
    return str(value)


def arm_table(arms: dict[str, dict[str, Path]]) -> list[str]:
    """One row per published checkpoint, every field read from its manifest."""
    lines = [
        "| Archivo | Backbone | Modo | Presupuesto (h-GPU) | Semilla | Pasos |",
        "|---|---|---|---|---|---|",
    ]
    for arm in sorted(arms):
        run = _read(arms[arm]["run"])
        execution = run.get("execution") or {}
        lines.append(
            f"| `{arms[arm]['weights'].name}` | {_fmt(run.get('backbone'))} | "
            f"{_fmt(run.get('adaptation_mode'))} | {_fmt(run.get('budget_gpu_hours'))} | "
            f"{_fmt(run.get('seed'))} | {_fmt(execution.get('steps_executed'))} |"
        )
    return lines


def _label_free(arms: dict[str, dict[str, Path]]) -> bool:
    return all(_read(entry["run"]).get("label_free") is True for entry in arms.values())


def render_notes(tag: str, arms: dict[str, dict[str, Path]], checksums: str) -> str:
    label_free = _label_free(arms)
    pretraining = (
        "preentrenados sin etiquetas" if label_free
        else "preentrenados (revisar `label_free` en los manifiestos)"
    )
    lines = [
        f"# Backbones adaptados al tono - {tag}",
        "",
        f"Extractores congelados, {pretraining} sobre dermatoscopias.",
        "Como cargarlos y que preprocesamiento usar: `docs/23-modelo-descargable-y-uso.md`.",
        "",
        *arm_table(arms),
        "",
        "Los arms con modo `paired` son el metodo propuesto (pares de tono);",
        "los demas son el control entrenado con el mismo presupuesto.",
        "",
        "| Archivo | Que es |",
        "|---|---|",
        "| `*_run.json` | Evidencia de cada corrida: presupuesto, semilla, hash de particion |",
        "| `bank.json`, `strata.json`, `bank.md` | Resultados del banco congelado |",
        "",
        "Comprobacion rapida de que el archivo descargado sirve:",
        "",
        "```bash",
        f"python scripts/try_model.py --backbone {sorted(arms)[0]}{WEIGHTS_SUFFIX} --image lesion.jpg",
        "```",
        "",
        "NO es un dispositivo de diagnostico: sin validacion clinica, entrenado",
        "sobre un conjunto mayoritariamente de piel clara y con peor desempeno",
        "en el tercil de piel menos luminosa.",
        "",
        "## Checksums",
        "",
        "```",
        checksums.rstrip("\n"),
        "```",
        "",
        "_Generado desde los `run.json` por `scripts/build_release_notes.py`._",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tag", required=True, help="Release tag")
    parser.add_argument("--dir", type=Path, required=True, help="Staging directory")
    parser.add_argument("--checksums", type=Path, help="SHA256SUMS file to embed")
    parser.add_argument("--out", type=Path, help="Where to write the notes (default: stdout)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        arms = discover_arms(args.dir)
    except MissingEvidence as exc:
        raise SystemExit(f"no publicable: {exc}")

    checksums = args.checksums.read_text(encoding="utf-8") if args.checksums else ""
    notes = render_notes(args.tag, arms, checksums)
    if args.out:
        args.out.write_text(notes, encoding="utf-8")
        print(f"notas escritas en {args.out} ({len(arms)} checkpoints)")
    else:
        print(notes, end="")


if __name__ == "__main__":
    main()
