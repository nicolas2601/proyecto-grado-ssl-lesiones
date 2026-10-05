"""FITZPATRICK.md (docs/33 E5), rendered from the per-dataset E5 JSON. Spanish, decimal comma.

Every number is read from a JSON; the words around them (glossary, caveats) are constants of
``e5_document`` and this module. A dataset whose JSON does not exist yet renders as ``pendiente``
with no number. A JSON that is a smoke run, claims to be validated, or carries a caveat different
from the code's is refused (``StaleDocumentError``): the table never publishes a document that
contradicts the limits it states.
"""

from __future__ import annotations

import json
from pathlib import Path

from dermssl import e5_document as doc
from dermssl import e5_ita as ita

DATASETS = {"ham10000": "HAM10000", "isic2019": "ISIC 2019", "isic2020": "ISIC 2020"}
TERCILE_ES = {"dark": "oscuro", "medium": "medio", "light": "claro", "unavailable": "sin L\\*"}
BUCKET_ES = {"abstain": "abstención"}
NA = "n/d"
ESTIMATOR_ES = {"otsu": "ITA, lesión por Otsu (referencia)", "whole_image": "ITA, imagen completa",
                "roi": "ITA, anillo perilesional (máscara)"}
PENDING = ("pendiente: este dataset todavía no se ha corrido (`scripts/e5_fitzpatrick.py` en el pod); "
           "no hay ningún número hasta que exista su JSON.")
GLOSSARY = [
    "- **ITA (ángulo de tipología individual):** ángulo calculado de la mediana de L\\* y b\\* de la piel en "
    "CIELAB, `arctan((L\\* − 50) / b\\*)`. Es una medida de color, no de fototipo.",
    "- **Bucket I-VI:** el tipo Fitzpatrick que resulta de cortar el ITA con los umbrales fijos de Groh "
    "(`fitzpatrick.FST_THRESHOLDS`). **Es la salida de un estimador no validado, no el fototipo real.** Un "
    "valor justo en un umbral va al tipo más oscuro.",
    "- **Abstención:** la imagen no recibe bucket (columna «abstención») cuando no hay piel suficiente, la "
    "máscara es degenerada, b\\* queda bajo el mínimo o el archivo no se pudo leer. Nunca se fuerza un bucket.",
    "- **Estimador de Otsu (referencia):** quita la lesión con un umbral de Otsu y lee la piel del resto; es el "
    "mismo en los tres datasets, así que es el comparable.",
    "- **Estimador de imagen completa:** todos los píxeles cuentan, sin quitar la lesión.",
    "- **Anillo perilesional (ROI de Pau):** banda justo fuera de la máscara de la lesión, sin pelo, reflejos, "
    "píxeles saturados ni borde negro, con abstención. Solo existe donde hay máscaras (HAM10000, `student_r3`); "
    "en ISIC 2019 e ISIC 2020 **no aplica** y no se inventan máscaras.",
    "- **L\\*:** luminosidad CIELAB (0 negro, 100 blanco). **Tercil de L\\*:** el dataset se divide en tres "
    "tercios por la mediana de L\\* de su piel; es luminancia, **no es tono de piel**, y sus cortes son "
    "propios de cada dataset.",
    "- **Descriptivo:** solo conteos. No hay exactitud, ni intervalos, ni contrastes: no se compara contra "
    "ningún fototipo real.",
    "- **n/d:** no disponible o no aplicable.",
]
READING = [
    "- Los conteos por bucket describen lo que el estimador ve, no la composición de fototipos del dataset: "
    "contra fototipos asignados por dermatólogos acierta 22,9 % frente a 51,3 % de la baseline de mayoría en "
    "PAD-UFES-20 (PMC13459198; docs/20, docs/30).",
    "- Los terciles son relativos a cada dataset: «oscuro» en HAM10000 y «oscuro» en ISIC 2020 no son la misma "
    "luminancia, y ninguno es piel oscura.",
    "- Que dos estimadores coincidan (cuadro Otsu × anillo) no los valida: dos estimadores no validados pueden "
    "coincidir y equivocarse juntos. Es acuerdo, no exactitud.",
    "- Los splits son los que ya existen en cada dataset (no se unifican, docs/33, sección 1): HAM10000 la "
    "partición de C1 agrupada por `lesion_id`, ISIC 2019 `StratifiedKFold` y ISIC 2020 agrupada por "
    "`patient_id`; «val» es el primer fold de cada una.",
    "- E5 no cambia el resultado primario del banco (docs/33, sección 5).",
]


class StaleDocumentError(ValueError):
    """An E5 JSON that the table must not publish."""


def num(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def dec(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pct(part: int, whole: int) -> str:
    return NA if not whole else f"{dec(100 * part / whole)} %"


def _table(header: list[str], rows: list[list[str]], text_last: bool = False) -> list[str]:
    aligns = ["---"] + ["---:"] * (len(header) - 1)
    if text_last:
        aligns[-1] = "---"
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(aligns) + "|"]
    return [*lines, *("| " + " | ".join(r) + " |" for r in rows), ""]


def _cross(table: dict, row_title: str, columns, labels: dict) -> list[str]:
    header = [row_title, *(labels.get(c, c) for c in columns), "Total"]
    rows = [[labels.get(name, name), *(num(v[c]) for c in columns), num(sum(v.values()))]
            for name, v in table.items()]
    return _table(header, rows)


def _check(document: dict, name: str) -> None:
    if document.get("smoke_limit") is not None:
        raise StaleDocumentError(f"{name}: smoke run ({document['smoke_limit']} per split) cannot be published")
    if document.get("validated") is not False or document.get("descriptive") is not True:
        raise StaleDocumentError(f"{name}: must be descriptive and not validated")
    if document.get("caveat") != doc.CAVEAT:
        raise StaleDocumentError(f"{name}: its caveat differs from the one in e5_document.CAVEAT")


def load_documents(root: Path) -> dict[str, dict]:
    """The E5 JSON found in ``root`` by dataset key, checked; missing datasets are simply absent."""
    found = {}
    for key in DATASETS:
        path = Path(root) / f"e5_fitzpatrick_{key}.json"
        if path.is_file():
            found[key] = json.loads(path.read_text(encoding="utf-8"))
            _check(found[key], path.name)
    return found


def _facts(d: dict) -> list[str]:
    report, split = d["report"], d["report"]["estimators"]["otsu"]["bucket_x_split"]
    parts = ", ".join(f"{s} {num(sum(split[s].values()))}" for s in d["splits"])
    masks = d["masks"]
    mask_text = ("sin máscaras" if masks is None else f"`{masks['producer']}`, sha256 `{masks['masks_sha256'][:12]}`")
    return [f"- Imágenes: {num(report['n_images'])} ({parts}); ilegibles: {num(report['n_unreadable'])}.",
            f"- Máscaras: {mask_text}.", f"- Revisión de código: `{d['code_revision']}`.", ""]


def _coverage(estimators: dict) -> list[str]:
    rows = []
    for name, est in estimators.items():
        reasons = "; ".join(f"{s}: {num(n)}" for s, n in est["status_counts"].items() if s != "ok") or "ninguno"
        if not est["applicable"]:
            rows.append([ESTIMATOR_ES[name], "no aplica", NA, NA, NA, reasons])
            continue
        total = sum(est["bucket_counts"].values())
        miss = est["bucket_counts"][ita.ABSTAIN]
        rows.append([ESTIMATOR_ES[name], "sí", num(total - miss), num(miss), pct(miss, total), reasons])
    header = ["Estimador", "¿Aplica?", "Con bucket", "Abstención", "% abstención", "Motivos (estado: n)"]
    return _table(header, rows, text_last=True)


def _distribution(estimators: dict) -> list[str]:
    columns = (*ita.BUCKET_LABELS, ita.ABSTAIN)
    rows = []
    for name, est in estimators.items():
        if est["applicable"]:
            total = sum(est["bucket_counts"].values())
            rows.append([ESTIMATOR_ES[name], *(f"{num(est['bucket_counts'][c])} ({pct(est['bucket_counts'][c], total)})"
                                               for c in columns)])
    return _table(["Estimador", *(BUCKET_ES.get(c, c) for c in columns)], rows)


def _estimator_tables(est: dict, title: str) -> list[str]:
    cols = (*ita.BUCKET_LABELS, ita.ABSTAIN)
    return [f"### Clase × bucket ({title})", "", *_cross(est["class_x_bucket"], "Clase", cols, BUCKET_ES),
            f"### Bucket × split ({title})", "", *_cross(est["bucket_x_split"], "Split", cols, BUCKET_ES)]


def _agreement(roi: dict) -> list[str]:
    cols = (*ita.BUCKET_LABELS, ita.ABSTAIN)
    both, same = roi["both_estimated"], roi["same_bucket"]
    return ["### Otsu × anillo (conteos; filas = Otsu, columnas = anillo)", "",
            *_cross(roi["agreement_with_otsu"], "Otsu \\ anillo", cols, BUCKET_ES),
            f"Con bucket en ambos estimadores: {num(both)} imágenes; mismo bucket en {num(same)} "
            f"({pct(same, both)}). Es acuerdo entre dos estimadores no validados, no exactitud.", ""]


def _terciles(t: dict) -> list[str]:
    cuts = t["cuts"]
    if cuts["low"] is None:
        text = f"No hay L\\* suficiente para cortar terciles (n = {num(cuts['n'])})."
    else:
        text = (f"Cortes de L\\* propios de este dataset (n = {num(cuts['n'])} con L\\*): oscuro L\\* ≤ "
                f"{dec(cuts['low'], 2)}; medio hasta {dec(cuts['high'], 2)}; claro L\\* > {dec(cuts['high'], 2)}. "
                "Son luminancia, no tono de piel.")
    cols = (*ita.TERCILES, ita.UNAVAILABLE)
    return ["### Terciles de L\\* (luminancia)", "", text, "", "**Clase × tercil**", "",
            *_cross(t["class_x_tercile"], "Clase", cols, TERCILE_ES), "**Tercil × split**", "",
            *_cross(t["tercile_x_split"], "Split", cols, TERCILE_ES)]


def _dataset(key: str, d: dict | None) -> list[str]:
    head = [f"## {DATASETS[key]}", ""]
    if d is None:
        return [*head, PENDING, ""]
    est = d["report"]["estimators"]
    out = [*head, *_facts(d), "### Cobertura de los estimadores", "", *_coverage(est),
           "### Distribución de buckets por estimador (conteo y % del total)", "", *_distribution(est),
           *_estimator_tables(est["otsu"], "Otsu")]
    if est["roi"]["applicable"]:
        out += [*_estimator_tables(est["roi"], "anillo perilesional"), *_agreement(est["roi"])]
    else:
        out += ["### Anillo perilesional", "", f"No aplica: {d['estimators']['roi']['reason']}.", ""]
    return [*out, *_terciles(d["report"]["terciles"])]


def render(documents: dict[str, dict]) -> str:
    """The whole FITZPATRICK.md from the checked documents found so far."""
    lines = ["# Banco completo: Fitzpatrick descriptivo (E5)", "",
             "Generado por `scripts/build_fitzpatrick_table.py` desde los JSON de `resultados/banco_completo/e5/`. "
             "No editar a mano. Plan y reglas: `docs/33-banco-completo-multidataset.md` (E5).", "",
             f"> **Aviso.** {doc.CAVEAT}", "", "## Glosario", "", *GLOSSARY, "", "## Cómo leer esta tabla", "",
             *READING, ""]
    for key in DATASETS:
        lines += _dataset(key, documents.get(key))
    return "\n".join(lines).rstrip("\n") + "\n"
