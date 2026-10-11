"""Generate the v8 descriptive audit table from cases.jsonl; --check verifies freshness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(root: Path) -> str:
    status = json.loads((root / "status.json").read_text())
    cases = [json.loads(line) for line in (root / "cases.jsonl").read_text().splitlines()]
    if status.get("status") != "complete" or status["n"] != len(cases) or not cases:
        raise ValueError("Audit is incomplete or empty")
    ids = [case["image_id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated audit IDs")
    groups = {}
    for case in cases:
        for view in case["views"]:
            groups.setdefault(view["name"], []).append(view)
    lines = ["# Auditoría descriptiva de vistas SSL", "",
             "Cobertura y retención respecto a pseudomáscaras; no son métricas anatómicas.", "",
             "| Vista | n | Máscara fuente vacía | Recorte sin lesión | Retención media válida |",
             "|---|---:|---:|---:|---:|"]
    for name, views in groups.items():
        retained = [v["lesion_retained_fraction"] for v in views
                    if v["lesion_retained_fraction"] is not None]
        mean = f"{sum(retained) / len(retained):.4f}" if retained else "NA"
        empty = sum(v["source_mask_empty"] for v in views)
        absent = sum(v["lesion_absent"] and not v["source_mask_empty"] for v in views)
        lines.append(f"| {name} | {len(views)} | {empty} | {absent} | {mean} |")
    lines += ["", "## Galería", ""]
    lines += [f"- [{i}]({i}/views.png)" for i in ids]
    lines += ["", "Generado desde `cases.jsonl`. No se cargan diagnósticos ni máscaras expertas.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output, text = args.root / "README.md", render(args.root)
    if args.check:
        if not output.exists() or output.read_text() != text:
            raise SystemExit("Audit report is stale")
    else:
        output.write_text(text)


if __name__ == "__main__":
    main()
