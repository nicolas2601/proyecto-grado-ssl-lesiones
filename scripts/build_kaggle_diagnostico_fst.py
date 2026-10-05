"""Genera y pushea, POR DATASET, un kernel de Kaggle (tipo *script*,
CPU-only) que corre la comparativa ITA sin-mascara vs con-mascara sobre
HAM10000, ISIC 2019 o ISIC 2020 - un kernel y un CSV de salida por dataset,
no uno combinado.

Por que por dataset y no uno solo
----------------------------------
Aislamiento de fallas: si el kernel combinado se corta a mitad de camino
(limite de tiempo de sesion de Kaggle), se pierde todo. Separado, un
dataset que falla no se lleva puesto a los otros dos. Tambien evita que
ISIC 2019 en resolucion completa (mucho mas pesado que los otros dos)
consuma el presupuesto de tiempo de una sesion compartida.

Un kernel de Kaggle no puede importar de este repositorio, asi que este
script inlina el codigo real (via ``inspect.getsource``, nunca una copia a
mano - la misma garantia que ``scripts/build_notebook_v5.py`` le da al
notebook de entrenamiento):

  - ``dermssl.data_resolver.index_images`` - resuelve las imagenes bajo
    ``/kaggle/input`` sin asumir a que profundidad Kaggle monta cada
    dataset (se vio anidado en ``/kaggle/input/datasets/<owner>/<slug>/``
    en un kernel tipo "script") y dedupea por stem - HAM10000 en Kaggle
    trae las mismas imagenes dos veces bajo carpetas con distinto casing.
    Ya resuelto y probado en ``dermssl``, no hace falta reinventarlo.
  - ``dermssl.data_resolver.DATASET_HANDLES`` - los tres slugs de Kaggle
    correctos, la misma fuente de verdad que usa el notebook real de
    Nico (confirmado contra el kernel-metadata.json real de
    ``nicorghost/sslmodelo-liviano``) - incluido el ISIC 2019 RESIZED, no
    la version de resolucion completa.
  - Solo dos funciones de ``pretrain/lesion_guided.py``
    (``otsu_threshold``, ``lesion_mask``) - las unicas que ``fitzpatrick.py``
    usa. El resto de ese archivo es el sampler multi-crop de SSL (v6,
    "thesis novelty 3"), sin relacion con este diagnostico - inlinarlo
    entero metia codigo y comentarios ajenos (en ingles, sobre DINO) en un
    kernel que se supone que es solo sobre FST.
  - ``fitzpatrick.py`` entero.
  - Las funciones puras de ``scripts/diagnostico_fst_comparativa.py``
    (``build_row``, ``write_csv``, ``print_summary``) - su CLI/argparse no
    aplica dentro de un kernel.

No necesita GPU (``enable_gpu=False``): el ITA sin mascara y el fallback
Otsu son pura numpy.

    python scripts/build_kaggle_diagnostico_fst.py --user mariapaus --dataset all --dry-run
    python scripts/build_kaggle_diagnostico_fst.py --user mariapaus --dataset ham10000 --push
    python scripts/build_kaggle_diagnostico_fst.py --user mariapaus --dataset all --push --real
    python scripts/build_kaggle_diagnostico_fst.py --user mariapaus --dataset all --status

``--real`` escribe ``SMOKE = False`` en el script generado (corrida
completa); sin el flag se pushea la version smoke (pocas imagenes) para
validar el cableado antes de la corrida real. ``--dataset all`` (default)
itera los tres; pasar uno puntual para reintentar solo ese.
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dermssl import data_resolver, fitzpatrick  # noqa: E402
from dermssl.pretrain import lesion_guided  # noqa: E402
import diagnostico_fst_comparativa as dfc  # noqa: E402

KERNEL_SLUG_PREFIX = "diagnostico-fst"
DATASETS = tuple(data_resolver.DATASET_HANDLES)  # ("ham10000", "isic2019", "isic2020")
SMOKE_LIMIT = 200

DIAGNOSTICO_HELPER_NAMES = (
    "image_id", "build_row", "write_csv",
    "_bootstrap_ci_bucket_cambio", "_transition_matrix", "_bucket_cambio_rate", "_artifact_breakdown",
    "summary_dict", "write_summary_json", "print_summary",
)
LESION_GUIDED_FUNCTION_NAMES = ("otsu_threshold", "lesion_mask")

DRIVER_TEMPLATE = '''
# --- driver: corre las dos corridas sobre este dataset, un CSV de salida ---
DATASET_ID = "__DATASET_ID__"
SMOKE = __SMOKE__  # False = corrida real. Cambiar con --real en el builder, no a mano.
SMOKE_LIMIT = __SMOKE_LIMIT__

INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")

images = index_images([INPUT_ROOT])
paths = sorted(images.values())
if SMOKE:
    paths = paths[:SMOKE_LIMIT]
print(f"{DATASET_ID}: {len(paths)} imagenes ({'smoke' if SMOKE else 'completo'})")
if not paths:
    raise SystemExit(f"no se encontro ninguna imagen bajo {INPUT_ROOT} para {DATASET_ID}")

rows = []
for i, path in enumerate(paths):
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    whole = estimate_fst_whole_image(img)
    masked = estimate_fst(img)
    rows.append(build_row(DATASET_ID, path, whole, masked))
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1}/{len(paths)}")

out_csv = WORKING / f"diagnostico_fst_{DATASET_ID}.csv"
write_csv(rows, out_csv)
summary = print_summary(DATASET_ID, rows, smoke=SMOKE)
out_json = WORKING / f"diagnostico_fst_{DATASET_ID}.json"
write_summary_json(summary, out_json)
print(f"escribi {len(rows)} filas -> {out_csv}")
print(f"escribi el resumen -> {out_json}")
'''


def strip_future_import(text: str) -> str:
    lines = [ln for ln in text.splitlines()
             if ln.strip() != "from __future__ import annotations"]
    return "\n".join(lines).strip()


def lesion_guided_shim() -> str:
    """Solo ``otsu_threshold`` y ``lesion_mask`` - las unicas dos funciones
    de ``lesion_guided.py`` que ``fitzpatrick.py`` usa. Se arma como modulo
    sintetico para que ``lesion_guided.lesion_mask(...)`` siga resolviendo
    igual que en el repo (``fitzpatrick.py`` hace
    ``from dermssl.pretrain import lesion_guided``, y un kernel de Kaggle
    no tiene el paquete ``dermssl`` instalado)."""
    src = "import numpy as np\n\n\n" + "\n\n\n".join(
        strip_future_import(inspect.getsource(getattr(lesion_guided, name)))
        for name in LESION_GUIDED_FUNCTION_NAMES
    )
    return (
        "import types as _types\n"
        "lesion_guided = _types.ModuleType('lesion_guided')\n"
        f"exec(compile({src!r}, 'lesion_guided.py', 'exec'), lesion_guided.__dict__)\n"
    )


def fitzpatrick_source() -> str:
    text = strip_future_import(inspect.getsource(fitzpatrick))
    lines = [ln for ln in text.splitlines()
             if ln.strip() != "from dermssl.pretrain import lesion_guided"]
    return "\n".join(lines)


def index_images_source() -> str:
    """``dermssl.data_resolver.index_images`` tal cual - resuelve rutas
    anidadas y dedupea por stem, ya probado, no reinventarlo aca."""
    return strip_future_import(inspect.getsource(data_resolver.index_images))


def diagnostico_helpers_source() -> str:
    parts = [
        f"MANIFEST_COLUMNS = {dfc.MANIFEST_COLUMNS!r}",
        f"N_BOOTSTRAP = {dfc.N_BOOTSTRAP!r}",
        f"BOOTSTRAP_SEED = {dfc.BOOTSTRAP_SEED!r}",
        "",
    ]
    parts += [inspect.getsource(getattr(dfc, name)) for name in DIAGNOSTICO_HELPER_NAMES]
    return "\n\n".join(parts)


def driver_source(dataset_id: str, real: bool) -> str:
    return (
        DRIVER_TEMPLATE
        .replace("__DATASET_ID__", dataset_id)
        .replace("__SMOKE__", repr(not real))
        .replace("__SMOKE_LIMIT__", repr(SMOKE_LIMIT))
    )


def build_script(dataset_id: str, real: bool = False) -> str:
    if dataset_id not in DATASETS:
        raise ValueError(f"dataset_id debe ser uno de {DATASETS}, no {dataset_id!r}")
    parts = [
        "# Generado por scripts/build_kaggle_diagnostico_fst.py - no editar a mano.",
        f"# Corrida A/B del ITA (sin mascara vs con mascara) sobre {dataset_id}.",
        "",
        "import csv",
        "import json",
        "import random",
        "from collections import Counter",
        "from pathlib import Path",
        "from typing import Iterable",
        "",
        "import numpy as np",
        "from PIL import Image",
        "",
        index_images_source(),
        "",
        lesion_guided_shim(),
        "",
        fitzpatrick_source(),
        "",
        diagnostico_helpers_source(),
        "",
        driver_source(dataset_id, real),
    ]
    return "\n".join(parts)


def kernel_slug(dataset_id: str) -> str:
    return f"{KERNEL_SLUG_PREFIX}-{dataset_id}".replace("_", "-")


def metadata(user: str, dataset_id: str) -> dict:
    return {
        "id": f"{user}/{kernel_slug(dataset_id)}",
        "title": kernel_slug(dataset_id),
        "code_file": "diagnostico_fst.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [data_resolver.DATASET_HANDLES[dataset_id]],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def stage(user: str, dataset_id: str, real: bool, staging: Path) -> Path:
    folder = staging / dataset_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "diagnostico_fst.py").write_text(build_script(dataset_id, real), encoding="utf-8")
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata(user, dataset_id), indent=2) + "\n", encoding="utf-8"
    )
    return folder


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--user", required=True, help="usuario de Kaggle dueno de los kernels")
    p.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    p.add_argument("--staging", default=str(REPO / ".kaggle-staging" / KERNEL_SLUG_PREFIX))
    p.add_argument("--real", action="store_true", help="SMOKE = False (corrida completa)")
    p.add_argument("--push", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status", action="store_true")
    args = p.parse_args()

    targets = DATASETS if args.dataset == "all" else (args.dataset,)

    for dataset_id in targets:
        slug = f"{args.user}/{kernel_slug(dataset_id)}"

        if args.status:
            subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", slug], check=False)
            continue

        folder = stage(args.user, dataset_id, args.real, Path(args.staging))
        mode = "REAL (SMOKE=False)" if args.real else "smoke"
        print(f"staged {slug} [{mode}] -> {folder}")

        if not args.push or args.dry_run:
            continue

        subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(folder)],
            check=True,
        )

    if args.push and not args.dry_run and not args.status:
        print(f"\ncheck progress:  python scripts/build_kaggle_diagnostico_fst.py "
              f"--status --user {args.user} --dataset {args.dataset}")


if __name__ == "__main__":
    main()
