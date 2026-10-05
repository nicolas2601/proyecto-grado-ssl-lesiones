"""Spanish text shared by every bench table: glossary, reading rules and per-dataset facts."""

from __future__ import annotations

from pathlib import Path

from dermssl.bench import metrics
from dermssl.bench.rows import V71_JSON, Dataset, _load

CLASSES_ES = {"benign": "benigno", "malignant": "maligno"}
GLOSSARY = [
    "- **Accuracy (exactitud):** porcentaje total de aciertos. Engaña con desbalance: si una clase domina, "
    "decir siempre esa clase ya da un número alto (ver abajo).",
    "- **Recall (sensibilidad):** de las lesiones que de verdad son de una clase, qué fracción detecta el "
    "modelo. **Recall macro** = el promedio de las clases (= exactitud balanceada).",
    "- **Precisión:** de las veces que el modelo dice «esta clase», qué fracción acierta. "
    "**Precisión macro** = el promedio de las clases.",
    "- **F1-score:** media armónica de precisión y recall; solo es alto si los dos lo son. **F1 macro** = "
    "el F1 de cada clase promediado sin pesar por tamaño. Es la **métrica principal** y el criterio de orden.",
    "- **AUC (OvR macro):** probabilidad de que el modelo le dé más puntaje de una clase a una imagen de esa "
    "clase que a una que no lo es (promedio de las clases). 0,5 = azar, 1 = perfecto. No depende de umbral.",
    "- **ECE (↓):** cuánto se aleja la confianza del modelo de su acierto real. Menor es mejor.",
    "- **MCC (coeficiente de correlación de Matthews):** resume toda la matriz de confusión en un número de "
    "-1 a 1 (0 = azar, 1 = perfecto). Para los baselines se calcula exacto desde `matriz_confusion`.",
    "- **IC 95 %:** intervalo de confianza por bootstrap, tal como lo publica cada fuente. Si los IC de dos "
    "modelos se solapan, no se puede afirmar que uno gane: es empate.",
    "- **Congelado (sin fine-tuning):** la red no se toca; solo se entrena el clasificador lineal de encima "
    "(regresión logística).",
    "- **Fine-tuning:** se reentrena toda la red con las etiquetas del dataset.",
    "- **Segmentación:** uso de máscaras de lesión como entrada. No se usa en este banco: la columna dice «no» "
    "en todas las filas (las variantes con segmentación son el bloque exploratorio E3 de docs/33).",
    "- **Transductivo:** el pool de preentrenamiento auto-supervisado contenía las imágenes de val (sin "
    "etiquetas). Solo aplica a B71/A71 originales en ISIC 2019 e ISIC 2020.",
]


def dataset_facts(repo: Path, ds: Dataset) -> dict:
    """n of val, split text, class order and majority-class share, read from the committed baseline JSON."""
    data = _load(repo, f"{ds.baselines}/results_dinov2_vitb14{ds.suffix}.json")
    probe = data["probes"]["logreg"]
    classes = list(probe["recall_por_clase"])
    supports = [sum(row) for row in probe["matriz_confusion"]]
    top = max(range(len(classes)), key=supports.__getitem__)
    return {"n_val": data["n_val"], "split": data["split"], "classes": classes,
            "majority": CLASSES_ES.get(classes[top], classes[top]), "majority_share": supports[top] / sum(supports),
            "cm": probe["matriz_confusion"]}


def macro_note(facts: dict) -> str:
    share = f"{facts['majority_share'] * 100:.1f}".replace(".", ",")
    return (f"En este val la clase «{facts['majority']}» es el {share} % de las imágenes "
            f"({len(facts['classes'])} clases): predecir siempre «{facts['majority']}» da una accuracy de "
            f"{facts['majority_share']:.4f}".replace(".", ",") + " y una exactitud balanceada de "
            f"{1 / len(facts['classes']):.4f}".replace(".", ",") + ".")


def _n(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def partitions(repo: Path, datasets: dict[str, Dataset]) -> list[str]:
    """Bullet list of the val partitions of the bench (they are not the same split)."""
    lines = []
    for ds in datasets.values():
        facts = dataset_facts(repo, ds)
        lines.append(f"- {ds.title}, baselines (Kaggle): val n = {_n(facts['n_val'])}; {facts['split']}.")
        if ds.key == "ham10000":
            lines.append(f"- {ds.title}, propuesto (v6/v7): val n = {_n(_load(repo, V71_JSON)['n_eval'])}; "
                         "partición agrupada por `lesion_id`.")
    return lines


def reading_rules(repo: Path, datasets: dict[str, Dataset]) -> list[str]:
    return [
        "Particiones de val de este banco (no son la misma partición):", "",
        *partitions(repo, datasets), "",
        "- **Entre particiones distintas la comparación es solo de intervalos (IC 95 %)**, no un test pareado: "
        "HAM10000 usa la partición v6/v7 (n = 2.004) para el propuesto y la de Kaggle (n = 2.014) para los "
        "baselines; ISIC 2019 e ISIC 2020 tienen la suya. Los contrastes pareados se harán solo donde dos "
        "modelos comparten partición e imágenes (docs/33, sección 5).",
        "- **El fine-tuning elige su mejor época sobre val** en las filas ya commiteadas (HAM10000 e "
        "ISIC 2019): su número es optimista. La corrida de ISIC 2020 (pendiente) elige en el holdout interno "
        "(20 % del train).",
        "- Los IC solo existen donde la fuente los publica: los baselines, para exactitud balanceada y F1 "
        "macro; B71/A71, para accuracy, exactitud balanceada, F1 macro, AUC y ECE. Precisión macro, recall "
        "macro y MCC de los baselines salen exactos de la matriz de confusión y van sin IC; en B71/A71 salen "
        "de las probabilidades commiteadas, también sin IC.",
        "- Las filas «pendiente» y «no ejecutado todavia» no tienen número: se listan para que nada parezca "
        "olvidado. No se inventa ninguna cifra.",
    ]
