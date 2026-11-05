"""Wait for both follow-up GPU queues to finish, then write the per-class table and a summary.

Detached companion of ssl_followup_runner_20261029.py; reads only, never trains.
"""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results/ssl-followup-20261029"
PYTHON = "/home/jovyan/work/proyectonp/.venv/bin/python"
KNOWN = [
    "SimSiam (3 sem.)=baseline-s42-pipeline-evaluation-v1,baseline-s43-evaluation-v1,baseline-s44-evaluation-v1",
    "VICReg (3 sem.)=vicreg-evaluation-v1,vicreg-s43-evaluation-v1,vicreg-s44-evaluation-v1",
    "Barlow+VICReg (3 sem.)=barlow-vicreg-evaluation-v1,barlow-vicreg-s43-evaluation-v1,barlow-vicreg-s44-evaluation-v1",
    "DINO+iBOT s42=dino-ibot-evaluation-v1",
]


def terminal_rows():
    rows = []
    for gpu in (0, 1):
        marker = STATE / f"gpu{gpu}/ALL_TERMINAL.json"
        if not marker.exists():
            return None
        rows.extend(json.loads(marker.read_text())["arms"])
    return rows


def registry_rows():
    rows = []
    for gpu in (0, 1):
        path = STATE / f"gpu{gpu}/registry.json"
        if path.exists():
            rows.extend(json.loads(path.read_text())["rows"])
    return rows


def summary(arms, registry):
    lines = ["# Seguimiento SSL 2026-10-29: resumen automático\n",
             "| Brazo | Estado | Sección | PAD F1 | Δ PAD (IC) | HAM F1 | Δ HAM (IC) |", "|---|---|---|---|---|---|---|"]
    by_id = {row["internal"]: row for row in registry}
    for arm in arms:
        row = by_id.get(arm["id"])
        if row:
            pad, ham = row["delta_pad"], row["delta_ham"]
            lines.append(f"| {arm['label']} | {arm['status']} | {arm['section']} | {row['pad_macro_f1']:.3f} | "
                         f"{pad[0]:+.3f} [{pad[1]:+.3f}, {pad[2]:+.3f}] | {row['ham_macro_f1']:.3f} | "
                         f"{ham[0]:+.3f} [{ham[1]:+.3f}, {ham[2]:+.3f}] |")
        else:
            lines.append(f"| {arm['label']} | {arm['status']} | {arm['section']} | - | - | - | - |")
            if arm.get("error"):
                lines.append(f"|  | error: {arm['error']} | | | | | |")
    lines.append("\nLos brazos 'guiado por etiquetas' usan etiquetas diagnósticas para elegir imágenes; no son SSL puro.")
    return "\n".join(lines) + "\n"


def main():
    while (arms := terminal_rows()) is None:
        time.sleep(60)
    done = [a for a in arms if a["status"] == "done"]
    groups = KNOWN + [f"{a['label']}={a['id']}-evaluation-v1" for a in done]
    subprocess.run([PYTHON, "scripts/per_class_report_20261029.py", str(STATE / "per-class-final"), *groups],
                   cwd=ROOT, check=False, timeout=900, capture_output=True)
    (STATE / "FINAL_SUMMARY.md").write_text(summary(arms, registry_rows()))
    (STATE / "FINISHER_DONE.json").write_text(json.dumps({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                          "arms": len(arms), "done": len(done)}) + "\n")


if __name__ == "__main__":
    main()
