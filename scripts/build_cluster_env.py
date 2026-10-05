"""Render resultados/entorno/cluster_env.json and docs/34 from the measured inputs; ``--check`` for CI.

    python scripts/build_cluster_env.py [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV = REPO / "resultados" / "entorno"
DOC = REPO / "docs" / "34-entorno-cluster.md"
DATASETS = {
    "ham10000": ("HAM10000", "v7 val n = 2.004, agrupado por lesion_id", "ver docs/28 §4.1"),
    "isic2019": ("ISIC 2019", "StratifiedKFold(5, shuffle, 42), primer fold", "8 clases, sin grupo real"),
    "isic2020": ("ISIC 2020", "StratifiedGroupKFold por patient_id (sklearn 1.7.2)", "binaria"),
}


def _load(name: str) -> dict:
    return json.loads((ENV / name).read_text(encoding="utf-8"))


def build_env() -> dict:
    hardware, pre = _load("hardware.json"), _load("transductividad_preflight.json")
    freeze = (ENV / "pip_freeze.txt").read_text(encoding="utf-8").splitlines()
    data = {k: {"n_images": v["n_ids"], "split": v["split_desc"],
                "val_n": v.get("val_baseline_skf_n", v.get("val_finetune_sgkf_n")),
                "val_sha256": v.get("val_baseline_skf_sha256", v.get("val_finetune_sgkf_sha256"))}
            for k, v in pre["datasets"].items()}
    return {"hardware_software": hardware, "pip_freeze_file": "resultados/entorno/pip_freeze.txt",
            "pip_freeze_lines": len(freeze), "datasets": data,
            "ssl_pool": {k: pre[k] for k in ("pool_n", "excluded_v7_n", "effective_pool_n", "exclude_ids_sha256_run",
                                              "clean_exclusion_n", "clean_effective_pool_n", "clean_exclude_file_sha256")}}


def render_doc(env: dict) -> str:
    hw, pool = env["hardware_software"], env["ssl_pool"]
    lines = [
        "# 34. Entorno del clúster y datos", "",
        "Generado por `scripts/build_cluster_env.py` desde `resultados/entorno/`; no editar a mano. Sin hostnames internos, IPs ni tokens.", "",
        "## Hardware", "",
        f"- GPU: 2 × {hw['gpus'][0]['model']}, {hw['gpus'][0]['vram_mib']} MiB cada una; driver {hw['nvidia_driver']}; CUDA {hw['cuda_torch_build']} (build de torch); cuDNN {hw['cudnn']}; NCCL {hw['nccl']}.",
        f"- CPU: {hw['cpu']['model']}, {hw['cpu']['logical_cpus']} CPU lógicas ({hw['cpu']['sockets']} socket, {hw['cpu']['threads_per_core']} hilos por núcleo).",
        f"- RAM: {hw['ram_gib']} GiB. `/dev/shm`: {hw['storage'][1]['size']} (los DataLoader usan 0 workers).",
        f"- Disco de datos: volumen de red compartido, {hw['storage'][0]['size']} ({hw['storage'][0]['used']} usados).",
        f"- SO: {hw['os']}, kernel {hw['kernel']}. Contenedor: {hw['container']}.", "",
        "## Software", "",
        f"- Python {hw['python']}, torch {hw['torch']}. scikit-learn por defecto {hw['sklearn_default']}; para reconstruir splits existentes se usa {hw['sklearn_for_existing_splits']}.",
        f"- `timm` en el venv: {'sí' if hw['timm_installed'] else 'no'}; para el fine-tuning: {hw['timm_for_finetune']}. Revisión de los re-preentrenamientos: `{hw['pretrain_revision']}`.",
        f"- Hilos: probe `PROBE_BLAS_THREADS` = {hw['thread_env']['PROBE_BLAS_THREADS']}; pretrain OMP/MKL/OPENBLAS = {hw['thread_env']['pretrain_OMP_MKL_OPENBLAS']}.",
        f"- `pip freeze` completo: `{env['pip_freeze_file']}` ({env['pip_freeze_lines']} líneas).", "",
        "## Datos", "",
        "| Dataset | Imágenes | Split de evaluación | val n | sha256 de la lista de val |", "|---|---|---|---|---|",
    ]
    for key, d in env["datasets"].items():
        name = DATASETS[key][0]
        lines.append(f"| {name} | {d['n_images']} | {d['split'].split(' - ')[0]} | {d['val_n']} | `{d['val_sha256']}` |")
    lines += [f"| HAM10000 | 10.015 | {DATASETS['ham10000'][1]} | 2.004 | ver docs/28 |", "",
              "## Pool de preentrenamiento SSL", "",
              f"- Pool v7: {pool['pool_n']} imágenes; exclusión v7: {pool['excluded_v7_n']} ids (sha256 `{pool['exclude_ids_sha256_run']}`); pool efectivo {pool['effective_pool_n']}.",
              f"- Pool limpio: exclusión {pool['clean_exclusion_n']} ids (sha256 `{pool['clean_exclude_file_sha256']}`); pool efectivo {pool['clean_effective_pool_n']}.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    env = build_env()
    outputs = {ENV / "cluster_env.json": json.dumps(env, indent=2, ensure_ascii=False) + "\n", DOC: render_doc(env)}
    for path, text in outputs.items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                print(f"{path} desactualizado: regenerar con scripts/build_cluster_env.py", file=sys.stderr)
                return 1
        else:
            path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
