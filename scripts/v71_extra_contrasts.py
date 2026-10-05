"""Extra paired contrasts on the committed v7.1 val probabilities (no new read of val).

    python scripts/v71_extra_contrasts.py [--out resultados/clasificacion/v71/extra_contrasts.json]

Reads only ``v71_final_eval.probas.npz`` and the v7 s43 probabilities; every
contrast is a paired bootstrap with the published seed (``dermssl.v71_extra``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl import v71_extra as vx  # noqa: E402
from dermssl import v71_metrics as vm  # noqa: E402
from dermssl import v71_plan as plan  # noqa: E402

V71 = REPO / "resultados" / "clasificacion" / "v71"
V7_S43 = REPO / "resultados" / "clasificacion" / "v7" / "s43" / "final_eval.probas.npz"
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
MEL, NV = CLASSES.index("mel"), CLASSES.index("nv")
V71_LABELS = plan.VAL_LABELS
S43_ROLES = {"A_v7_s43": "plain", "B_v7_s43": "proposed"}
FAIR_PAIRS = (("A71", "public_A71"), ("B71", "public_B71"), ("A71", "public71"), ("B71", "public71"))
MEL_TO_NV_PAIRS = (("A71", "public_A71"), ("B71", "public_B71"), ("A71", "A_v7"), ("B71", "B_v7"))
S43_PAIRS = (("A71", "A_v7"), ("A71", "A_v7_s43"), ("A71", "B_v7_s43"),
             ("B71", "B_v7"), ("B71", "B_v7_s43"), ("B71", "A_v7_s43"))
S43_METRICS = ("macro_f1", "balanced_accuracy")

NOTE_FAIR = ("Cada lectura se evalua con el probe entrenado en el 80 % del fit y con SU PROPIO umbral de "
             "holdout (el protocolo publicado); el contraste pareado usa el mismo remuestreo para ambas.")
NOTE_MEL_TO_NV = ("Melanomas verdaderos cuyo argmax es nv (probe del train completo), sobre el total de mel "
                  "verdaderos. Menor es mejor: un delta negativo favorece al brazo.")
NOTE_S43 = ("A* v7.1 es la sopa de los pesos s42 y s43, asi que su contraste contra A* v7 s43 no es "
            "independiente de s43; B* v7.1 usa solo s42. Las filas contra A_v7/B_v7 son s42 y repiten "
            "las publicadas.")
NOTE_VERDICT = "'mejora' solo si el IC pareado excluye el 0; nunca 'gana' si el IC toca el 0."


def check_aligned(v71, s43) -> None:
    if not np.array_equal(v71["y_true"], s43["y_true"]):
        raise ValueError("y_true de v7.1 y de v7 s43 no coinciden: un contraste pareado exige las mismas imagenes")


def load_inputs(v71_dir: Path, s43_path: Path) -> dict:
    v71 = np.load(v71_dir / "v71_final_eval.probas.npz")
    s43 = np.load(s43_path)
    check_aligned(v71, s43)
    doc = json.loads((v71_dir / "v71_final_eval.json").read_text(encoding="utf-8"))
    full = {label: v71[label] for label in V71_LABELS} | {k: s43[role] for k, role in S43_ROLES.items()}
    return {"y_true": v71["y_true"], "full": full,
            "fit80": {label: v71[f"{label}__fit80"] for label in V71_LABELS},
            "thresholds": {label: doc["results"][label]["holdout_thresholds"] for label in V71_LABELS},
            "sources": [v71_dir / "v71_final_eval.probas.npz", v71_dir / "v71_final_eval.json", s43_path]}


def fair_mel_section(inputs: dict) -> dict:
    return {"note": NOTE_FAIR, "contrasts": vx.fair_operating_contrasts(
        inputs["y_true"], inputs["fit80"], inputs["thresholds"], FAIR_PAIRS, mel=MEL)}


def mel_to_nv_section(inputs: dict) -> dict:
    y = inputs["y_true"]
    return {"note": NOTE_MEL_TO_NV,
            "entries": {k: vx.mel_to_nv(y, p, mel=MEL, nv=NV) for k, p in inputs["full"].items()},
            "contrasts": vx.argmax_contrasts(y, inputs["full"], MEL_TO_NV_PAIRS, (vx.MEL_TO_NV,), mel=MEL, nv=NV)}


def s43_section(inputs: dict) -> dict:
    return {"note": NOTE_S43,
            "contrasts": vx.argmax_contrasts(inputs["y_true"], inputs["full"], S43_PAIRS, S43_METRICS)}


def document(inputs: dict) -> dict:
    return {
        "split": "val", "n_eval": len(inputs["y_true"]), "classes": list(CLASSES),
        "sources": {str(p.relative_to(REPO)): rp.file_sha256(p) for p in inputs["sources"]},
        "bootstrap": {"n_resamples": vm.BOOTSTRAP_RESAMPLES, "rng_seed": vm.BOOTSTRAP_SEED, "paired": True},
        "note_verdict": NOTE_VERDICT,
        "fair_mel": fair_mel_section(inputs),
        "mel_to_nv": mel_to_nv_section(inputs),
        "s43": s43_section(inputs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=V71 / "extra_contrasts.json")
    args = parser.parse_args(argv)
    doc = document(load_inputs(V71, V7_S43))
    args.out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    run_environment.write_sidecar(args.out, args)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
