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

Transports (``--via``):
  kaggle    the official CLI: ``pip install kaggle`` + ~/.kaggle/kaggle.json
  composio  ``composio proxy`` with the Kaggle account linked in Composio -
            no local token file needed (default when the CLI is absent).
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
    "isic2020": ["nischaydnk/isic-2020-jpg-256x256-resized"],
    # The RESIZED ISIC 2019 only ships a binary target; the eight classes of the
    # official challenge - the ones Gessert's 0.636 anchor is measured on - are
    # only in this release.
    "isic2019": ["andrewmvd/isic-2019"],
}

# SDK enum: NvidiaTeslaT4 | NvidiaTeslaP100. NvidiaTeslaT4x2 does not exist and
# Kaggle silently ignores it.
MACHINE_SHAPE = "NvidiaTeslaT4"

BANKS = {
    "baselines": ("ham10000", "test_{m}.ipynb"),
    "baselines_isic2020": ("isic2020", "test_{m}_isic2020.ipynb"),
    "baselines_isic2019": ("isic2019", "test_{m}_isic2019.ipynb"),
    "finetune": ("ham10000", "finetune_{m}.ipynb"),
    "finetune_isic2020": ("isic2020", "finetune_{m}_isic2020.ipynb"),
    "finetune_isic2019": ("isic2019", "finetune_{m}_isic2019.ipynb"),
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
        # Kaggle's PyTorch build ships no sm_60 kernels, so a Tesla P100 dies on
        # the first forward with "no kernel image is available". Both transports
        # must pin the T4: the CLI reads this key, composio reads push_payload().
        "machine_shape": MACHINE_SHAPE,
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


KAGGLE_API = "https://www.kaggle.com/api/v1"


def push_payload(folder: Path) -> dict:
    """Body of POST /kernels/push, built from the staged kernel-metadata.json."""
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    return {
        "slug": meta["id"],
        "newTitle": meta["title"],
        "text": (folder / meta["code_file"]).read_text(),
        "language": meta["language"],
        "kernelType": meta["kernel_type"],
        "isPrivate": meta["is_private"],
        "enableGpu": meta["enable_gpu"],
        "enableTpu": meta["enable_tpu"],
        "enableInternet": meta["enable_internet"],
        "datasetDataSources": meta["dataset_sources"],
        "competitionDataSources": meta["competition_sources"],
        "kernelDataSources": meta["kernel_sources"],
        "modelDataSources": meta["model_sources"],
        "categoryIds": [],
        "machineShape": meta.get("machine_shape", MACHINE_SHAPE),
    }


def composio_proxy(url: str, method: str = "GET", body: dict | None = None) -> dict:
    cmd = ["composio", "proxy", url, "--toolkit", "kaggle", "-X", method]
    if body is not None:
        cmd += ["-H", "content-type: application/json", "-d", "-"]
    out = subprocess.run(cmd, input=json.dumps(body) if body is not None else None,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"composio proxy failed: {out.stderr.strip() or out.stdout.strip()}")
    text = out.stdout.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def push_via_composio(folder: Path) -> dict:
    return composio_proxy(f"{KAGGLE_API}/kernels/push", "POST", push_payload(folder))


def status_via_composio(slug: str) -> dict:
    user, name = slug.split("/", 1)
    return composio_proxy(f"{KAGGLE_API}/kernels/status?userName={user}&kernelSlug={name}")


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
    p.add_argument("--via", choices=("auto", "kaggle", "composio"), default="auto")
    args = p.parse_args(argv)

    via = args.via
    if via == "auto":
        via = "kaggle" if shutil.which("kaggle") else "composio"
    if via == "composio" and shutil.which("composio") is None:
        sys.exit("neither kaggle nor composio CLI found")

    for model in args.models:
        slug = f"{args.user}/{kernel_slug(args.bank, model)}"
        if args.status:
            if via == "composio":
                st = status_via_composio(slug)
                print(f"{slug}: {st.get('status', st)} {st.get('failureMessage', '')}".strip())
            else:
                subprocess.run(["kaggle", "kernels", "status", slug], check=False)
            continue
        folder = stage(args.bank, model, args.user, args.real, Path(args.staging))
        mode = "REAL (SMOKE=False)" if args.real else "smoke"
        if args.dry_run:
            print(f"staged {slug} [{mode}] -> {folder}")
            continue
        print(f"pushing {slug} [{mode}] via {via} ...")
        if via == "composio":
            resp = push_via_composio(folder)
            if resp.get("error") or "ref" not in resp:
                sys.exit(f"push failed for {slug}: {resp}")
            print(f"  ok -> {resp.get('url') or resp['ref']} (version {resp.get('versionNumber')})")
        else:
            subprocess.run(["kaggle", "kernels", "push", "-p", str(folder)], check=True)
    if not args.status and not args.dry_run:
        print("\ncheck progress:  python scripts/kaggle_push.py --status "
              f"--bank {args.bank} --user {args.user} --via {via}")
        print("fetch outputs:   kaggle kernels output <user>/<slug> -p results/")


if __name__ == "__main__":
    main()
