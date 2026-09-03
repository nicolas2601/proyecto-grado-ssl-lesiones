"""Push the generated Kaggle notebooks with the Kaggle CLI, bank by bank.

Writes a kernel-metadata.json next to each notebook (dataset sources and GPU
settings included) and runs ``kaggle kernels push`` so the notebook is
uploaded AND executed (Save & Run All) without touching the browser.

    python scripts/kaggle_push.py --bank baselines_isic2020 --user nicorghost --dry-run
    python scripts/kaggle_push.py --bank baselines_isic2020 --user nicorghost --real
    python scripts/kaggle_push.py --bank finetune --user nicorghost --models resnet50 vit_b16
    python scripts/kaggle_push.py --status --bank baselines_isic2020 --user nicorghost

``--real`` flips SMOKE = False inside a copy of the notebook before pushing;
without it the smoke version is pushed (validate the wiring first).
Requires ``pip install kaggle`` and ~/.kaggle/kaggle.json (a NEW token).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dermssl.finetune import protocol  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

DATASET_SOURCES = {
    "ham10000": ["kmader/skin-cancer-mnist-ham10000"],
    "isic2019": ["nischaydnk/isic-2019-jpg-224x224-resized"],
    "isic2020": ["nischaydnk/isic-2020-jpg-256x256-resized"],
}

BANKS = {
    "baselines": ("ham10000", "test_{m}.ipynb"),
    "baselines_isic2019": ("isic2019", "test_{m}_isic2019.ipynb"),
    "baselines_isic2020": ("isic2020", "test_{m}_isic2020.ipynb"),
    "finetune": ("ham10000", "finetune_{m}.ipynb"),
    "finetune_isic2019": ("isic2019", "finetune_{m}_isic2019.ipynb"),
    "finetune_isic2020": ("isic2020", "finetune_{m}_isic2020.ipynb"),
}


def kernel_slug(bank: str, model: str) -> str:
    return f"{bank}-{model}".replace("_", "-")


def metadata(user: str, bank: str, model: str, code_file: str) -> dict:
    dataset = BANKS[bank][0]
    return {
        "id": f"{user}/{kernel_slug(bank, model)}",
        "title": kernel_slug(bank, model),
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": DATASET_SOURCES[dataset],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def set_smoke(notebook_path: Path, smoke: bool) -> dict:
    nb = json.loads(notebook_path.read_text())
    hits = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for i, line in enumerate(cell["source"]):
            if line.startswith("SMOKE = "):
                cell["source"][i] = f"SMOKE = {smoke}" + line[line.index("  #"):] \
                    if "  #" in line else f"SMOKE = {smoke}\n"
                hits += 1
    assert hits == 1, f"expected one SMOKE line in {notebook_path}, found {hits}"
    return nb


def stage(bank: str, model: str, user: str, real: bool, staging: Path) -> Path:
    dataset, pattern = BANKS[bank]
    src = REPO / "notebooks" / bank / model / pattern.format(m=model)
    if not src.is_file():
        raise FileNotFoundError(src)
    dest = staging / kernel_slug(bank, model)
    dest.mkdir(parents=True, exist_ok=True)
    nb = set_smoke(src, smoke=not real)
    (dest / src.name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    (dest / "kernel-metadata.json").write_text(
        json.dumps(metadata(user, bank, model, src.name), indent=2) + "\n")
    return dest


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", required=True, choices=tuple(BANKS))
    p.add_argument("--user", required=True, help="Kaggle username (owner of the kernels)")
    p.add_argument("--models", nargs="*", default=list(protocol.MODELS))
    p.add_argument("--real", action="store_true", help="SMOKE = False (full run)")
    p.add_argument("--dry-run", action="store_true", help="stage files, do not push")
    p.add_argument("--status", action="store_true", help="only query kernel status")
    p.add_argument("--staging", default="/tmp/kaggle-staging")
    args = p.parse_args(argv)

    if not args.status and not args.dry_run and shutil.which("kaggle") is None:
        sys.exit("kaggle CLI not found: pip install kaggle (and ~/.kaggle/kaggle.json)")

    for model in args.models:
        slug = f"{args.user}/{kernel_slug(args.bank, model)}"
        if args.status:
            subprocess.run(["kaggle", "kernels", "status", slug], check=False)
            continue
        folder = stage(args.bank, model, args.user, args.real, Path(args.staging))
        mode = "REAL (SMOKE=False)" if args.real else "smoke"
        if args.dry_run:
            print(f"staged {slug} [{mode}] -> {folder}")
            continue
        print(f"pushing {slug} [{mode}] ...")
        subprocess.run(["kaggle", "kernels", "push", "-p", str(folder)], check=True)
    if not args.status and not args.dry_run:
        print("\ncheck progress:  python scripts/kaggle_push.py --status "
              f"--bank {args.bank} --user {args.user}")
        print("fetch outputs:   kaggle kernels output <user>/<slug> -p results/")


if __name__ == "__main__":
    main()
