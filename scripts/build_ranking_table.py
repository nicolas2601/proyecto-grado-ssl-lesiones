"""Build resultados/clasificacion/RANKING_MACRO_F1.md from committed JSON.

One table ranks every classifier the repo has measured on HAM10000 val by
macro-F1, best to worst: frozen baselines, supervised fine-tuning, the v6
bank, the v7 seeds and the v7.1 arms. Nothing is transcribed; ``--check``
fails if the markdown drifts from the JSON.

    python scripts/build_ranking_table.py            # write
    python scripts/build_ranking_table.py --check    # CI
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINES = Path("results") / "baselines"
FINETUNE = Path("results") / "finetune" / "ham10000"
LRPROBE = Path("results") / "finetune_lrprobe"
CLS = Path("resultados") / "clasificacion"
BANK = CLS / "bank.json"
V7 = {42: CLS / "v7" / "final_eval.json", 43: CLS / "v7" / "s43" / "final_eval.json",
      44: CLS / "v7" / "s44" / "final_eval.json"}
V71 = CLS / "v71" / "v71_final_eval.json"
TARGET = REPO / CLS / "RANKING_MACRO_F1.md"
OURS = "propio"
V6_ARM = {"probe": "logreg", "fraction": 1.0, "seed": 0}
V6_BACKBONES = {"ssl_plain_vitb14": "v6 SSL sin término de tono (ViT-B/14)",
                "ssl_proposed_vitb14": "v6 SSL propuesto (ViT-B/14)"}
V7_ROLES = {"plain": "A* (SSL sin término de tono)", "proposed": "B* (SSL propuesto)"}
V71_LABELS = {"A71": "A71 (sopa A* s42+s43, cls_attn, la0.5, TTA)",
              "B71": "B71 (B* s42, cls4, balanced, TTA)",
              "public71": "DINOv2 público, su mejor protocolo v7.1",
              "public_B71": "DINOv2 público con el protocolo de B71",
              "public_A71": "DINOv2 público con el protocolo de A71"}
MODEL_NAMES = {"alexnet": "AlexNet", "vgg19": "VGG19", "resnet50": "ResNet-50",
               "efficientnet_b0": "EfficientNet-B0", "vit_b16": "ViT-B/16",
               "simsiam_r50": "SimSiam ResNet-50", "dinov2_vits14": "DINOv2 ViT-S/14",
               "dinov2_vitb14": "DINOv2 ViT-B/14"}
NO_CI = "—"


@dataclass(frozen=True)
class Entry:
    name: str
    family: str
    macro_f1: float
    ci: tuple[float, float] | None
    split: str
    source: Path


def _read(path: Path) -> dict:
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def _ci(values: list[float] | None) -> tuple[float, float] | None:
    return (values[0], values[1]) if values else None


def _model(data: dict) -> str:
    return MODEL_NAMES.get(data["modelo"], data["modelo"])


def _kaggle_split(data: dict) -> str:
    return f"Kaggle, val n = {data['n_val']}"


def frozen_entries() -> list[Entry]:
    """Frozen baselines: the probe is the one the baselines README picks (best balanced accuracy)."""
    out = []
    for path in sorted((REPO / BASELINES).glob("results_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("smoke"):
            continue
        probes = data["probes"]
        probe = max(probes, key=lambda name: probes[name]["balanced_accuracy"])
        rec = probes[probe]
        out.append(Entry(f"{_model(data)} congelado + {probe}", "congelado (pesos públicos)",
                         rec["macro_f1"], _ci(rec["ic95_bootstrap"]["macro_f1"]), _kaggle_split(data),
                         path.relative_to(REPO)))
    return out


def finetune_entries() -> list[Entry]:
    out = []
    for path in sorted((REPO / FINETUNE).glob("results_*_ft.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rec = data["probes"]["finetune"]
        out.append(Entry(f"{_model(data)} fine-tuning", "fine-tuning supervisado", rec["macro_f1"],
                         _ci(rec["ic95_bootstrap"]["macro_f1"]), _kaggle_split(data),
                         path.relative_to(REPO)))
    return out


def lrprobe_entries() -> list[Entry]:
    """Fine-tuning re-run with a smaller backbone LR, kept out of the main ranking."""
    out = []
    for path in sorted((REPO / LRPROBE).glob("results_*_lrprobe.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rec = data["probes"]["finetune"]
        ci = rec.get("ic95_bootstrap", {}).get("macro_f1")
        lr = data["protocolo"]["lr_backbone"]
        out.append(Entry(f"{_model(data)} fine-tuning, LR backbone {lr:g}", "fine-tuning supervisado",
                         rec["macro_f1"], _ci(ci), _kaggle_split(data), path.relative_to(REPO)))
    return out


def v6_entries() -> list[Entry]:
    bank = _read(BANK)
    split = f"v6/v7, val n = {bank['split']['val']}"
    rows = {r["backbone"]: r for r in bank["rows"] if all(r[k] == v for k, v in V6_ARM.items())}
    return [Entry(label, OURS, rows[b]["macro_f1"], None, split, BANK) for b, label in V6_BACKBONES.items()]


def v7_entries() -> list[Entry]:
    out = []
    for seed, path in V7.items():
        doc = _read(path)
        split = f"v6/v7, val n = {doc['n_eval']}"
        for role, label in V7_ROLES.items():
            rec = doc["results"][role]
            out.append(Entry(f"{label}, semilla {seed}", OURS, rec["macro_f1"],
                             _ci(rec["ci95"]["macro_f1"]), split, path))
    base = _read(V7[42])["results"]["public"]
    out.append(Entry("DINOv2 público, probe v7 base", "congelado (pesos públicos)", base["macro_f1"],
                     _ci(base["ci95"]["macro_f1"]), f"v6/v7, val n = {_read(V7[42])['n_eval']}", V7[42]))
    return out


def v71_entries() -> list[Entry]:
    doc = _read(V71)
    split = f"v6/v7, val n = {doc['n_eval']}"
    out = []
    for key, label in V71_LABELS.items():
        rec = doc["results"][key]
        family = "congelado (pesos públicos)" if key.startswith("public") else OURS
        out.append(Entry(label, family, rec["macro_f1"], _ci(rec["ci95"]["macro_f1"]), split, V71))
    return out


def ranked() -> list[Entry]:
    entries = [*frozen_entries(), *finetune_entries(), *v6_entries(), *v7_entries(), *v71_entries()]
    return sorted(entries, key=lambda e: e.macro_f1, reverse=True)


def _num(x: float) -> str:
    return f"{x:.4f}".replace(".", ",")


def _cell(entry: Entry) -> str:
    if entry.ci is None:
        return f"{_num(entry.macro_f1)} {NO_CI}"
    return f"{_num(entry.macro_f1)} [{_num(entry.ci[0])}; {_num(entry.ci[1])}]"


def _overlaps(a: Entry, b: Entry) -> bool:
    return a.ci is not None and b.ci is not None and a.ci[0] <= b.ci[1] and b.ci[0] <= a.ci[1]


def _versus(entry: Entry, best: Entry) -> str:
    """Interval comparison against the top row; descriptive, not a paired test."""
    if entry is best:
        return "referencia (1.º)"
    if entry.ci is None or best.ci is None:
        return "sin IC"
    return "IC se solapa" if _overlaps(entry, best) else "IC separado"


def _table(entries: list[Entry], best: Entry | None) -> list[str]:
    header = "| # | Modelo | Familia | Macro-F1 [IC 95 %] | Frente al 1.º | Val | Fuente |"
    lines = [header, "|---|---|---|---|---|---|---|"]
    for i, e in enumerate(entries, 1):
        versus = _versus(e, best) if best else NO_CI
        lines.append(f"| {i} | {e.name} | {e.family} | {_cell(e)} | {versus} | {e.split} | "
                     f"`{e.source.as_posix()}` |")
    return lines


def _best_supervised(entries: list[Entry]) -> Entry:
    return next(e for e in entries if e.family == "fine-tuning supervisado")


def _verdict(entries: list[Entry]) -> list[str]:
    best, sup = entries[0], _best_supervised(entries)
    beaten = [e.name for e in entries if e.family != OURS and e.ci and best.ci and not _overlaps(e, best)]
    tie = "se solapan" if _overlaps(best, sup) else "no se solapan"
    return [
        "## Lectura", "",
        f"- 1.º: {best.name}, {_cell(best)}.",
        f"- Mejor supervisado: {sup.name}, {_cell(sup)}. Los IC del 1.º y del mejor supervisado "
        f"{tie}: {'empate estadístico, no victoria' if _overlaps(best, sup) else 'diferencia por IC'}.",
        f"- Modelos externos con IC separado del 1.º ({len(beaten)}): {', '.join(beaten)}.",
    ]


CAVEATS = [
    "## Cómo leer esta tabla", "",
    "- Métrica: macro-F1 sobre HAM10000 val (mayor es mejor). IC 95 % bootstrap de cada fuente; "
    f"«{NO_CI}» = la fuente no publica IC.",
    "- Hay dos particiones de val, ambas agrupadas por `lesion_id`: la de Kaggle (baselines y "
    "fine-tuning) y la de v6/v7. Entre particiones las filas no son pareadas; «Frente al 1.º» "
    "compara intervalos, no es un test pareado. DINOv2 ViT-B/14 público congelado sirve de ancla: "
    "aparece en ambas.",
    "- El fine-tuning supervisado elige su mejor época sobre val, así que su número es optimista.",
    "- Parte de la ganancia de v7.1 viene del protocolo del probe (TTA, pooling, ajuste de logits), "
    "no del backbone: compare A71 y B71 con «DINOv2 público con el protocolo de …».",
]


def render() -> str:
    entries = ranked()
    lr = sorted(lrprobe_entries(), key=lambda e: e.macro_f1, reverse=True)
    return "\n".join([
        "# Ranking por macro-F1 — todos los clasificadores de HAM10000", "",
        "Generado por `scripts/build_ranking_table.py` desde los JSON del repositorio. "
        "No editar a mano.", "",
        *CAVEATS, "",
        "## Ranking (mejor a peor)", "",
        *_table(entries, entries[0]), "",
        *_verdict(entries), "",
        "## Fuera del ranking: fine-tuning con LR de backbone reducido", "",
        "Re-corrida exploratoria con otra tasa de aprendizaje; no forma parte del banco registrado.", "",
        *_table(lr, None), "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the markdown is stale")
    args = parser.parse_args()
    text = render()
    if args.check:
        if TARGET.read_text(encoding="utf-8") != text:
            print(f"{TARGET.relative_to(REPO)} is stale: run scripts/build_ranking_table.py", file=sys.stderr)
            return 1
        return 0
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
