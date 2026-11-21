#!/usr/bin/env python3
"""Compute-cost tables for the v10.4.2 record, rendered from the benchmark evidence (EN or ES).

  python build_compute_tables.py --evidence DIR [--lang en|es] [--readme README.md]

Reads bench_cluster_cpu.json (CPU thread scaling), bench_kaggle.json (GPU, optional),
latency_vps_segmenter.json and vps_analyze_baseline.json. Prints Markdown, or with --readme rewrites the
block between the compute-tables markers of that file.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BLOCK = re.compile(r"(<!-- compute-tables:start[^>]*-->\n).*?(<!-- compute-tables:end -->)", re.S)

MODELS = ["classifier_v10_4_0_d4", "classifier_v10_5_0_single", "segmenter_v10_4_1_resnet34", "segmenter_v10_4_2_dpt"]
THREADS = [2, 4, 8, 16]
CONFIGS = {  # (classifier, segmenter)
    "fast": ("classifier_v10_4_0_d4", "segmenter_v10_4_1_resnet34"),
    "full": ("classifier_v10_4_0_d4", "segmenter_v10_4_2_dpt"),
    "single_full": ("classifier_v10_5_0_single", "segmenter_v10_4_2_dpt"),
}
TEXT = {
    "en": {
        "names": {"classifier_v10_4_0_d4": "Classifier v10.4.0 (D4, 8 views, 448 px)",
                  "classifier_v10_5_0_single": "Classifier v10.5.0 (1 view, 448 px)",
                  "segmenter_v10_4_1_resnet34": "Segmenter v10.4.1 (ResNet34 U-Net, 224 px)",
                  "segmenter_v10_4_2_dpt": "Segmenter v10.4.2 (DINOv2-B + DPT, 448 px)"},
        "configs": {"fast": "Production today (fast): v10.4.0 + segmenter v10.4.1",
                    "full": "Full: v10.4.0 + segmenter v10.4.2",
                    "single_full": "Single view + new segmenter: v10.5.0 + segmenter v10.4.2"},
        "t1": "Per-model inference (ONNX Runtime, batch 1, model only; p50 ms)",
        "h1": ["Model", "File MB", "Peak RAM MB", "CPU 2 thr", "CPU 4 thr", "CPU 8 thr", "CPU 16 thr", "GPU p50 ms",
               "GPU memory MB"],
        "t2": "Per analysis by configuration (classifier + segmenter, model time only; seconds)",
        "h2": ["Configuration", "CPU 2 thr", "CPU 4 thr", "CPU 8 thr", "CPU 16 thr", "GPU", "Model RAM MB"],
        "t3": "Measured on the production VPS (2 vCPU host, container 1.5 CPU, 2 threads)",
        "h3": ["Stage", "Today (fast)", "With segmenter v10.4.2"],
        "stages": ["Classification", "Segmentation", "Photometry", "Total (sum of stages)", "Container RAM"],
        "hw": "Hardware",
        "na": "not measured",
        "notes": "Model time excludes image decoding and pre/post-processing; the VPS table includes them. RAM is the "
                 "peak resident memory of one process that loads only that model (each includes the ONNX Runtime "
                 "baseline, so configuration sums slightly overstate a single process).",
    },
    "es": {
        "names": {"classifier_v10_4_0_d4": "Clasificador v10.4.0 (D4, 8 vistas, 448 px)",
                  "classifier_v10_5_0_single": "Clasificador v10.5.0 (1 vista, 448 px)",
                  "segmenter_v10_4_1_resnet34": "Segmentador v10.4.1 (ResNet34 U-Net, 224 px)",
                  "segmenter_v10_4_2_dpt": "Segmentador v10.4.2 (DINOv2-B + DPT, 448 px)"},
        "configs": {"fast": "Produccion hoy (rapida): v10.4.0 + segmentador v10.4.1",
                    "full": "Completa: v10.4.0 + segmentador v10.4.2",
                    "single_full": "Vista unica + segmentador nuevo: v10.5.0 + segmentador v10.4.2"},
        "t1": "Inferencia por modelo (ONNX Runtime, lote 1, solo el modelo; p50 en ms)",
        "h1": ["Modelo", "Archivo MB", "RAM pico MB", "CPU 2 hilos", "CPU 4 hilos", "CPU 8 hilos", "CPU 16 hilos",
               "GPU p50 ms", "Memoria GPU MB"],
        "t2": "Por analisis segun configuracion (clasificador + segmentador, solo tiempo de modelo; segundos)",
        "h2": ["Configuracion", "CPU 2 hilos", "CPU 4 hilos", "CPU 8 hilos", "CPU 16 hilos", "GPU", "RAM modelos MB"],
        "t3": "Medido en el VPS de produccion (host de 2 vCPU, contenedor 1,5 CPU, 2 hilos)",
        "h3": ["Etapa", "Hoy (rapida)", "Con segmentador v10.4.2"],
        "stages": ["Clasificacion", "Segmentacion", "Fotometria", "Total (suma de etapas)", "RAM del contenedor"],
        "hw": "Hardware",
        "na": "no medido",
        "notes": "El tiempo de modelo no incluye decodificar la imagen ni el pre/posprocesamiento; la tabla del VPS si "
                 "los incluye. La RAM es el pico residente de un proceso que carga solo ese modelo (cada uno incluye la "
                 "base de ONNX Runtime, asi que las sumas por configuracion sobrestiman un poco un solo proceso).",
    },
}


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def index(report: dict | None, provider: str) -> dict:
    if report is None:
        return {}
    return {(r["model"], r["threads"] if provider == "cpu" else 0): r for r in report["results"]
            if r.get("provider") == provider and "p50_ms" in r}


def fmt(value: float | None, digits: int = 0, na: str = "-") -> str:
    return na if value is None else f"{value:,.{digits}f}"


def model_rows(t: dict, cpu: dict, gpu: dict) -> list[list[str]]:
    rows = []
    for key in MODELS:
        base = cpu.get((key, THREADS[0]), {})
        g = gpu.get((key, 0), {})
        rows.append([t["names"][key], fmt(base.get("model_mb"), 1), fmt(base.get("peak_rss_mb"))]
                    + [fmt(cpu.get((key, n), {}).get("p50_ms"), 1, t["na"]) for n in THREADS]
                    + [fmt(g.get("p50_ms"), 1, t["na"]), fmt(g.get("gpu_mem_delta_mb"), 0, t["na"])])
    return rows


def config_rows(t: dict, cpu: dict, gpu: dict) -> list[list[str]]:
    rows = []
    for name, pair in CONFIGS.items():
        cells = []
        for n in THREADS:
            parts = [cpu.get((m, n), {}).get("p50_ms") for m in pair]
            cells.append(t["na"] if None in parts else fmt(sum(parts) / 1000, 2))
        g = [gpu.get((m, 0), {}).get("p50_ms") for m in pair]
        ram = [cpu.get((m, THREADS[0]), {}).get("peak_rss_mb") for m in pair]
        rows.append([t["configs"][name]] + cells + [t["na"] if None in g else fmt(sum(g) / 1000, 3),
                                                    t["na"] if None in ram else fmt(sum(ram))])
    return rows


def vps_rows(t: dict, baseline: dict, dpt: dict, cpu: dict) -> list[list[str]]:
    ms = baseline["timing_ms_last_call"]
    new_seg = dpt["p50_ms"]
    today = [ms["classification"], ms["segmentation"], ms["photometry"]]
    full = [ms["classification"], new_seg, ms["photometry"]]
    ram_today = baseline["container_memory_in_use_gib"] * 1024
    old_seg, new_seg_ram = (cpu[(m, THREADS[0])]["peak_rss_mb"]
                            for m in (CONFIGS["fast"][1], CONFIGS["full"][1]))
    ram_full = ram_today - old_seg + new_seg_ram  # swap the segmenter's own footprint (model-only processes)
    rows = [[t["stages"][i], f"{today[i] / 1000:.1f} s", f"{full[i] / 1000:.1f} s"] for i in range(3)]
    rows.append([t["stages"][3], f"{sum(today) / 1000:.1f} s", f"{sum(full) / 1000:.1f} s"])
    rows.append([t["stages"][4], f"{ram_today:,.0f} MB", f"~{ram_full:,.0f} MB"])
    return rows


def render(evidence: Path, lang: str) -> str:
    t = TEXT[lang]
    load = lambda name: json.loads((evidence / name).read_text()) if (evidence / name).is_file() else None  # noqa: E731
    cpu_report, gpu_report = load("bench_cluster_cpu.json"), load("bench_kaggle.json")
    cpu, gpu = index(cpu_report, "cpu"), index(gpu_report, "cuda")
    hw = [f"CPU: {cpu_report['hardware']['cpu']} ({cpu_report['hardware']['logical_cpus']} logical), "
          f"ONNX Runtime {cpu_report['hardware']['onnxruntime']}"]
    if gpu_report:
        hw.append(f"GPU: {gpu_report['hardware']['gpu']}, ONNX Runtime {gpu_report['hardware']['onnxruntime']}")
    parts = [f"**{t['hw']}:** " + "; ".join(hw), f"### {t['t1']}", table(t["h1"], model_rows(t, cpu, gpu)),
             f"### {t['t2']}", table(t["h2"], config_rows(t, cpu, gpu)),
             f"### {t['t3']}", table(t["h3"], vps_rows(t, load("vps_analyze_baseline.json"),
                                                        load("latency_vps_segmenter.json"), cpu)), t["notes"]]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--lang", choices=sorted(TEXT), default="en")
    parser.add_argument("--readme", type=Path)
    args = parser.parse_args()
    tables = render(args.evidence, args.lang)
    if args.readme is None:
        print(tables, end="")
        return
    text = args.readme.read_text()
    if not BLOCK.search(text):
        raise SystemExit(f"{args.readme}: compute-tables markers not found")
    args.readme.write_text(BLOCK.sub(lambda m: m.group(1) + "\n" + tables + "\n" + m.group(2), text, count=1))


if __name__ == "__main__":
    main()
