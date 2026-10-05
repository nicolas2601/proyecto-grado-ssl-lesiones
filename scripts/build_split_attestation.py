"""State what the provenance gaps are, and what the evidence does not prove.

Two arms of this run carry incomplete manifests, each for a known reason:
``ssl_plain_vits14`` recorded sixty-four zeros as its split_hash because the
orchestrator never passed the flag, and the manually launched arm recorded no
code_revision because the command did not export GIT_COMMIT.

Neither run.json is edited. A manifest rewritten after the fact stops being
evidence: it looks exactly like one that was correct all along, and nothing
in the file distinguishes them. The gaps are recorded here instead, beside
what was checked afterwards - and beside the limits of that check, which is
the part a reader needs and the part that goes missing first.

    python scripts/build_split_attestation.py --tone <salidas/tone> \\
        --pool-receipt <tone/inputs/pool_split_hash.json> --out <destino.json>
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

ZERO_HASH = "0" * 64
RUN_FILE = "run.json"


def read_arms(tone: Path) -> dict[str, dict]:
    """Every arm directory that already has a manifest."""
    arms = {}
    for manifest in sorted(tone.glob(f"*/{RUN_FILE}")):
        arms[manifest.parent.name] = json.loads(manifest.read_text(encoding="utf-8"))
    return arms


def gaps_for(arm: str, run: dict) -> list[str]:
    """The provenance fields this arm cannot answer for itself."""
    found = []
    if run.get("split_hash") == ZERO_HASH:
        found.append("split_hash en ceros: la corrida no registro sobre que datos entreno")
    elif not run.get("split_hash"):
        found.append("split_hash ausente")
    if not run.get("code_revision"):
        found.append("code_revision nulo: no consta con que version del codigo se entreno")
    if not run.get("data_revision"):
        found.append("data_revision nulo")
    return found


SAME_CODE_UNKNOWABLE = "no se puede saber"


def same_code_verdict(arms: dict[str, dict]) -> str:
    """Whether the arms can be shown to have run the same code.

    Three states, and the third one has to say what it says. Two arms with
    `code_revision: null` are not an agreement and not a disagreement - they
    are an absence of evidence, and a verdict of `false` would be read as
    "different" while `unknown` would be read as "not checked yet". Neither
    is true. The arms compared here were launched from the same checkout;
    that is knowledge from outside the artefacts, and it belongs in the
    prose, not in a field that looks like it was measured.
    """
    revisions = {arm: run.get("code_revision") for arm, run in arms.items()}
    known = {arm: rev for arm, rev in revisions.items() if rev}
    if len(known) < 2:
        return SAME_CODE_UNKNOWABLE
    if len(known) < len(revisions):
        return SAME_CODE_UNKNOWABLE
    return "si" if len(set(known.values())) == 1 else "no"


def build(arms: dict[str, dict], pool_hash: str, pool_count: int) -> dict:
    """The attestation: the gaps, the later check, and what it cannot settle."""
    per_arm = {}
    for arm, run in sorted(arms.items()):
        declared = run.get("execution", {}).get("dataset_image_count")
        per_arm[arm] = {
            "gaps": gaps_for(arm, run),
            "split_hash_declarado": run.get("split_hash"),
            "code_revision": run.get("code_revision"),
            "dataset_image_count": declared,
            "coincide_con_el_pool_actual": (
                None if declared is None else declared == pool_count
            ),
        }

    corroborated = [
        arm for arm, entry in per_arm.items()
        if entry["coincide_con_el_pool_actual"] and entry["gaps"]
    ]
    return {
        "fecha": date.today().isoformat(),
        "mismo_codigo_entre_brazos": same_code_verdict(arms),
        "pool_split_hash_recalculado": pool_hash,
        "pool_imagenes": pool_count,
        "brazos": per_arm,
        "que_se_verifico": (
            "El pool presente hoy tiene el hash indicado y esa cantidad de imagenes. "
            f"Los brazos {corroborated or 'ninguno'} declararon el mismo "
            "dataset_image_count, lo que es consistente con haber entrenado sobre "
            "este mismo pool."
        ),
        "sobre_el_codigo": (
            "Un veredicto de 'no se puede saber' significa que al menos un brazo "
            "no registro con que version del codigo entreno. NO significa que "
            "corrieran codigo distinto, ni que la comparacion no sirva: significa "
            "que esa garantia no sale del artefacto y tiene que sostenerse por "
            "fuera, diciendo como."
        ),
        "que_NO_prueba": (
            "Que el pool fuera identico durante el entrenamiento de los brazos con "
            "hueco. La verificacion es posterior: coincidir en la cantidad de "
            "imagenes es evidencia, no demostracion, porque un pool con las mismas "
            "55.034 imagenes cambiadas de contenido daria el mismo conteo y otro "
            "hash. Ningun run.json fue editado; los huecos se declaran, no se rellenan."
        ),
    }


def to_markdown(attestation: dict) -> str:
    lines = [
        "## Huecos de procedencia y que se verifico despues",
        "",
        f"Recalculado el {attestation['fecha']} sobre el pool presente:",
        f"`{attestation['pool_split_hash_recalculado'][:16]}...`, "
        f"{attestation['pool_imagenes']:,} imagenes.".replace(",", "."),
        "",
        "| Brazo | Hueco | dataset_image_count | Coincide con el pool |",
        "|---|---|---|---|",
    ]
    for arm, entry in attestation["brazos"].items():
        gaps = "; ".join(entry["gaps"]) or "ninguno"
        match = {True: "si", False: "NO", None: "sin dato"}[entry["coincide_con_el_pool_actual"]]
        lines.append(f"| `{arm}` | {gaps} | {entry['dataset_image_count'] or '-'} | {match} |")
    lines += ["", f"**Que NO prueba.** {attestation['que_NO_prueba']}"]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tone", type=Path, required=True, help="salidas/tone directory")
    parser.add_argument("--pool-receipt", type=Path, required=True,
                        help="tone/inputs/pool_split_hash.json")
    parser.add_argument("--pool-count", type=int, required=True,
                        help="Recursive .png count of the pool")
    parser.add_argument("--out", type=Path, required=True, help="Attestation JSON to write")
    parser.add_argument("--markdown", type=Path, help="Optional human-readable copy")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    arms = read_arms(args.tone)
    if not arms:
        raise SystemExit(f"no hay ningun {RUN_FILE} bajo {args.tone}")
    pool_hash = json.loads(args.pool_receipt.read_text(encoding="utf-8"))["split_hash"]

    attestation = build(arms, pool_hash, args.pool_count)
    args.out.write_text(json.dumps(attestation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(to_markdown(attestation), encoding="utf-8")

    with_gaps = [arm for arm, entry in attestation["brazos"].items() if entry["gaps"]]
    print(f"atestacion escrita en {args.out}; brazos con hueco: {with_gaps or 'ninguno'}")


if __name__ == "__main__":
    main()
