"""Generate reproducible FST diagnostic figures from repository JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "figuras_analisis"
DATASETS = (
    ("HAM10000", ROOT / "resultados_fst_v3_ham10000" / "diagnostico_fst_ham10000.json"),
    ("ISIC2019", ROOT / "resultados_fst_v3_isic2019" / "diagnostico_fst_isic2019.json"),
    ("ISIC2020", ROOT / "resultados_fst_v3_isic2020" / "diagnostico_fst_isic2020.json"),
)


def load_rows() -> list[dict[str, int | float | str]]:
    rows = []
    for name, path in DATASETS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": name,
                "images": int(payload["n"]),
                "comparables": int(payload["comparables"]),
                "bucket_change": int(payload["bucket_cambio"]),
                "bucket_change_pct": float(payload["bucket_cambio_pct"]),
                "degenerate": int(payload["status_con_mascara"].get("degenerate_mask", 0)),
            }
        )
    return rows


def write_svg(
    rows: list[dict[str, int | float | str]],
    key: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    width, height = 920, 520
    margin_left, margin_bottom = 100, 90
    plot_width, plot_height = width - margin_left - 40, height - margin_bottom - 70
    maximum = max(float(row[key]) for row in rows) or 1.0
    bar_width = plot_width / (len(rows) * 1.7)
    gap = bar_width * 0.7
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{title}</text>',
        f'<text x="22" y="{height / 2}" transform="rotate(-90 22 {height / 2})" text-anchor="middle" font-family="Arial" font-size="15">{ylabel}</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 40}" y2="{height - margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="55" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333"/>',
    ]
    for index, row in enumerate(rows):
        value = float(row[key])
        bar_height = value / maximum * plot_height
        x = margin_left + gap + index * (bar_width + gap)
        y = height - margin_bottom - bar_height
        elements.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#255f85"/>'
        )
        elements.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="14">{value:g}</text>'
        )
        elements.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - margin_bottom + 28}" text-anchor="middle" font-family="Arial" font-size="14">{row["dataset"]}</text>'
        )
    elements.append("</svg>")
    (OUTPUT / filename).write_text("\n".join(elements), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    write_svg(
        rows,
        "bucket_change_pct",
        "Cambio de bucket FST al usar máscara",
        "Porcentaje de casos comparables",
        "fst_cambio_bucket.svg",
    )
    write_svg(
        rows,
        "degenerate",
        "Máscaras degeneradas por dataset",
        "Casos",
        "fst_mascaras_degeneradas.svg",
    )


if __name__ == "__main__":
    main()

