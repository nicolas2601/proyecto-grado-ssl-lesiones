"""Assemble a single site_data.json for the public demo from the real sources.

The web demo must show every advance — segmentation, training, classification
with confidence bands, downloads — and must never invent a number. So the site
reads ONE generated file instead of hand-typed figures, and this script builds
that file from the committed sources:

  - resultados/clasificacion/bank.json        -> classification metrics
  - resultados/clasificacion/bootstrap.json   -> paired confidence intervals
  - resultados/clasificacion/strata.json      -> classification tone equity
  - resultados/clasificacion/runs/*/run.json  -> training budget per arm
  - plan-segmentacion/.../evaluacion_*/summary.json -> segmentation Dice/HD95
  - resultados/tono/strata_v3/summary_*.json  -> segmentation tone equity

A number the sources do not carry does not reach the page. Run:

    python scripts/build_site_data.py --out resultados/site_data.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLASIF = REPO / "resultados" / "clasificacion"
SEG = REPO / "plan-segmentacion" / "camino-b-cnn" / "resultados" / "2026-09-21-final"
TONO_SEG = REPO / "resultados" / "tono" / "strata_v3"

# Dataset counts live in the write-up, not a JSON the pipeline emits; carried
# here with an explicit source so the page can cite them and a reviewer can
# check them against resultados/README.md section 1.
DATASETS = {
    "pool_total": 58457,
    "pool_valid": 55034,
    "pool_valid_pct": 94.1,
    "quarantined": 3423,
    "pretraining": ["HAM10000", "ISIC 2019 (sin la parte de HAM)", "ISIC 2020"],
    "not_in_training": "ISIC 2018 (solo evaluacion externa de segmentacion)",
    "source": "resultados/README.md seccion 1",
}

DISCLAIMER = (
    "No es un dispositivo de diagnostico. Entrenado sobre dermatoscopia a 224px, "
    "sobre un pool mayoritariamente de piel clara y con peor desempeno en el tercil "
    "de piel menos luminosa. Una imagen de camara de celular es fuera de "
    "distribucion: las etiquetas que produzca son demostrativas, no clinicas."
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _seg_eval(name: str) -> dict:
    """The headline stats of one segmentation evaluation, from its summary."""
    data = _load(SEG / f"evaluacion_{name}" / "summary.json")
    m = data["metrics"]

    def stat(key: str) -> dict:
        s = m.get(key, {})
        return {k: s.get(k) for k in ("mean", "median", "ci95_low", "ci95_high", "p10", "n")}

    return {
        "n_images": data["meta"].get("n_images"),
        "resolution": data["meta"].get("resolution"),
        "dice": stat("dice"),
        "iou": stat("iou"),
        "hd95": stat("hd95"),
        "sensitivity": stat("sensitivity"),
        "specificity": stat("specificity"),
    }


def _seg_tone(name: str) -> dict:
    """Per-L*-tercile Dice equity of the segmenter (the corrected v3 strata)."""
    data = _load(TONO_SEG / f"summary_{name}.json")
    strata = data.get("strata", {})
    return {
        "buckets": strata.get("buckets", strata),
        "gap_dice": strata.get("gap_dice"),
        "worst_bucket": strata.get("worst_bucket"),
        "source": str((TONO_SEG / f"summary_{name}.json").relative_to(REPO)),
    }


def _training() -> list[dict]:
    arms = []
    for run_file in sorted(CLASIF.glob("runs/*/run.json")):
        r = _load(run_file)
        ex = r.get("execution") or {}
        arms.append({
            "arm": run_file.parent.name,
            "adaptation_mode": r.get("adaptation_mode"),
            "initial_weights": r.get("initial_weights"),
            "budget_gpu_hours": r.get("budget_gpu_hours"),
            "steps": ex.get("steps_executed"),
            "samples": ex.get("samples_processed"),
            "pairs": r.get("dataset_pair_count"),
        })
    return arms


def build() -> dict:
    bank = _load(CLASIF / "bank.json")
    bootstrap = _load(CLASIF / "bootstrap.json")
    strata = _load(CLASIF / "strata.json")
    return {
        "meta": {
            "title": "Segmentacion y clasificacion auto-supervisada de lesiones cutaneas",
            "repo": "https://github.com/nicolas2601/proyecto-grado-ssl-lesiones",
            "branch": "proposed-cluster",
            "release": "https://github.com/nicolas2601/proyecto-grado-ssl-lesiones/releases/tag/proposed-cluster-v1",
            "disclaimer": DISCLAIMER,
            "framing": (
                "Los cuatro brazos SSL parten de los pesos publicos de DINOv2 y "
                "adaptan al dominio con SimSiam; es 'DINOv2 + adaptacion de dominio', "
                "no un modelo entrenado desde cero."
            ),
        },
        "datasets": DATASETS,
        "segmentation": {
            "isic2018_external": _seg_eval("isic2018"),
            "ham10000_in_pool": _seg_eval("ham10000"),
            "tone_equity": {
                "isic2018": _seg_tone("isic2018"),
                "ham10000": _seg_tone("ham10000"),
            },
            "note": (
                "ISIC 2018 es externo (fuera del pool): es la cifra que habla de "
                "generalizacion. HAM10000 esta dentro del pool."
            ),
        },
        "training": {
            "arms": _training(),
            "note": (
                "Igual presupuesto de COMPUTO (12 h-GPU por brazo), no igual muestras: "
                "el brazo pareado lee dos imagenes por paso."
            ),
        },
        "classification": {
            "backbones": bank.get("backbones", {}),
            "rows": bank.get("rows", []),
            "headline_probe": "logreg",
            "strata_probe": "logreg_balanced",
            "split": bank.get("split", {}),
        },
        "classification_tone_equity": strata,
        "contrasts": {
            "n_resamples": bootstrap.get("n_resamples"),
            "rng_seed": bootstrap.get("rng_seed"),
            "paired": bootstrap.get("paired"),
            "note_ece": bootstrap.get("note_ece"),
            "results": bootstrap.get("results", []),
        },
        "downloads": {
            "release_tag": "proposed-cluster-v1",
            "backbones": [
                "ssl_plain_vitb14_backbone.pt",
                "ssl_plain_vits14_backbone.pt",
                "ssl_proposed_vitb14_backbone.pt",
                "ssl_proposed_vits14_backbone.pt",
            ],
            "checksums": "SHA256SUMS.txt",
            "note": "Cada backbone viaja con su run.json. El clasificador (probe) se publica aparte.",
        },
        "classifier": {
            "status": "pending",
            "note": (
                "El backbone extrae el vector; el probe (cabeza clasificadora) lo "
                "convierte en una de 7 clases. Se completa cuando el artefacto exista."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "resultados" / "site_data.json")
    args = parser.parse_args()
    data = build()
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"site_data.json escrito en {args.out}")
    # Fail loud if a whole section came back empty — a silent empty section is
    # the same absence-reads-as-success bug the rest of this repo guards against.
    empties = [k for k in ("segmentation", "training", "classification", "contrasts")
               if not data.get(k)]
    if empties:
        raise SystemExit(f"secciones vacias, revisar las fuentes: {empties}")


if __name__ == "__main__":
    main()
