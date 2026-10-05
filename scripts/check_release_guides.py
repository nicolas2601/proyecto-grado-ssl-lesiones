"""Run the usage guide literally in clean virtual environments against a tiny model.

For each form (a: torch, b: transformers, c: ONNX in Python) this creates a
fresh venv WITHOUT the repository, runs the ``guide-<form>-install`` block of
docs/USO_MODELOS.md line by line with that venv first on PATH, then runs the
``guide-<form>`` Python block unchanged inside a tiny release folder (random
DINOv2 in the release layout, model name ``B_s42``, image ``lesion.jpg``).
The printed probabilities must match the repository's own computation to
1e-4. A failing step means the guide is wrong: fix the guide, not the check.

    python scripts/check_release_guides.py --forms a b c --workdir ~/.cache/guide-check

Needs network (pip) and the facebookresearch/dinov2 source ($DINOV2_REPO or
the torch hub cache) to build the tiny model.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

GUIDE = REPO / "docs" / "USO_MODELOS.md"
MODEL = "B_s42"
IMAGE = "lesion.jpg"
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
BLOCK = re.compile(r"```(\w+) (guide-[\w-]+)\n(.*?)```", re.DOTALL)
TINY = dict(embed_dim=32, depth=2, num_heads=2)


def extract_blocks(text: str) -> dict:
    return {tag: code for _, tag, code in BLOCK.findall(text)}


def dinov2_source() -> Path | None:
    candidates = [os.environ.get("DINOV2_REPO")]
    try:
        import torch  # noqa: PLC0415

        candidates.append(str(Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"))
    except ImportError:
        pass
    found = [Path(c) for c in candidates if c and (Path(c) / "dinov2" / "models").is_dir()]
    return found[0] if found else None


def tiny_hub(source: Path, seed: int = 0, **overrides):
    """Random DinoVisionTransformer shaped like the hub's dinov2_vitb14, only smaller."""
    import torch  # noqa: PLC0415

    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from dinov2.models.vision_transformer import DinoVisionTransformer  # noqa: PLC0415

    torch.manual_seed(seed)
    kwargs = dict(img_size=518, patch_size=14, mlp_ratio=4, init_values=1.0, block_chunks=0,
                  num_register_tokens=0, interpolate_antialias=False, interpolate_offset=0.1, **TINY)
    model = DinoVisionTransformer(**{**kwargs, **overrides})
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(torch.randn_like(param) * (0.5 if "gamma" in name else 0.2))
    return model.eval()


def _lesion_image(path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:450, 0:600]
    blob = np.exp(-(((xx - 300) / 120.0) ** 2 + ((yy - 225) / 90.0) ** 2))
    rgb = np.stack([200 - 120 * blob, 160 - 110 * blob, 140 - 90 * blob], axis=-1)
    rgb += rng.normal(0, 8, rgb.shape)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path, quality=95)


def build_tiny_release(root: Path, source: Path) -> dict:
    """Tiny release folder; returns the reference probabilities computed with repo code."""
    import torch  # noqa: PLC0415

    import finalize_release  # noqa: PLC0415
    from dermssl.release import package, plain_dinov2, preprocess, torch_forms  # noqa: PLC0415

    rng = np.random.default_rng(3)
    weight, bias = rng.normal(size=(7, TINY["embed_dim"])) * 2, rng.normal(size=7)
    torch_forms.write_model_forms(tiny_hub(source, seed=9), weight, bias, CLASSES, root / MODEL,
                                  preprocess.PREPROCESSING, fp16=False)
    package.write_json(root / MODEL / "release.json", {"model": MODEL, "classes": CLASSES})
    finalize_release.copy_files(root, finalize_release.CODE_COPIES)
    _lesion_image(root / IMAGE)
    model, classes = plain_dinov2.load_classifier(root / MODEL / "hf")
    with torch.no_grad():
        proba = model(torch.from_numpy(preprocess.preprocess_batch([root / IMAGE])))[0]
    return dict(zip(classes, proba.double().tolist()))


def parse_output(stdout: str) -> dict:
    rows = [line.split("\t") for line in stdout.strip().splitlines() if "\t" in line]
    return {name: float(value) for name, value in rows}


def check_output(stdout: str, reference: dict, atol: float = 1e-4) -> None:
    got = parse_output(stdout)
    if set(got) != set(reference):
        raise AssertionError(f"guide printed classes {sorted(got)}, expected {sorted(reference)}")
    worst = max(abs(got[k] - reference[k]) for k in reference)
    if worst > atol:
        raise AssertionError(f"guide probabilities differ from the reference by {worst:.2e}")


def run_block(python: str, code: str, cwd: Path, env: dict | None = None) -> str:
    done = subprocess.run([python, "-c", code], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=900)
    if done.returncode != 0:
        raise AssertionError(f"guide block failed:\n{done.stderr[-3000:]}")
    return done.stdout


def _venv(workdir: Path, form: str, install: str) -> tuple[str, dict]:
    venv = workdir / f"venv-{form}"
    shutil.rmtree(venv, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    env.update(PATH=f"{venv / 'bin'}{os.pathsep}{env['PATH']}", VIRTUAL_ENV=str(venv))
    for line in filter(None, (raw.strip() for raw in install.splitlines())):
        print(f"[{form}] $ {line}", flush=True)
        subprocess.run(["bash", "-c", line], env=env, check=True, cwd=workdir)
    python = str(venv / "bin" / "python")
    leaked = subprocess.run([python, "-c", "import dermssl"], env=env, capture_output=True)
    if leaked.returncode == 0:
        raise AssertionError("the clean venv can import dermssl: the repository leaked in")
    return python, env


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--forms", nargs="+", default=["a", "b", "c"], choices=["a", "b", "c"])
    parser.add_argument("--workdir", type=Path, default=Path.home() / ".cache" / "guide-check")
    args = parser.parse_args(argv)
    source = dinov2_source()
    if source is None:
        print("ABORT: DINOv2 source not found; set DINOV2_REPO")
        return 1
    blocks = extract_blocks(GUIDE.read_text(encoding="utf-8"))
    root = args.workdir / "release"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    reference = build_tiny_release(root, source)
    for form in args.forms:
        python, env = _venv(args.workdir, form, blocks[f"guide-{form}-install"])
        check_output(run_block(python, blocks[f"guide-{form}"], root, env), reference)
        print(f"[{form}] guide OK in a clean venv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
