"""Assemble release/v7 after packaging and parity: docs, code, results, sizes, checksums.

Refuses a model without parity.json or with any failed gate (exact
reproduction of final_eval.json included; there is no override). Removes the
fp16 ONNX copy of a model that failed the registered fp16 rule, writes the
model card with the models' deployment metrics, and fails if any file exceeds
the 2 GiB GitHub asset limit. Nothing is uploaded.

    python scripts/finalize_release.py --release release/v7
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dermssl import v7_seeds as seeds  # noqa: E402
from dermssl.release import package  # noqa: E402

MODELS = seeds.model_names(("A", "B"))
RELEASE_SRC = REPO / "src" / "dermssl" / "release"
V7 = REPO / "resultados" / "clasificacion" / "v7"
_RESULTS = {f"results/s{seed}/{path.name}": REPO / path for seed in seeds.REGISTERED_SEEDS
            for path in (seeds.final_eval_path(seed), seeds.probas_path(seed))}
COPIES = {
    "README.md": REPO / "docs" / "release" / "README.md",
    "LICENSE": REPO / "LICENSE",
    "LICENSE-WEIGHTS": REPO / "LICENSE-WEIGHTS",
    "USO_MODELOS.md": REPO / "docs" / "USO_MODELOS.md",
    "preprocess.py": RELEASE_SRC / "preprocess.py",
    "dinov2_plain.py": RELEASE_SRC / "plain_dinov2.py",
    "web/index.html": REPO / "docs" / "release" / "web" / "index.html",
    "web/preprocess.js": REPO / "docs" / "release" / "web" / "preprocess.js",
    **_RESULTS,
    "results/SEMILLAS.md": V7 / "SEMILLAS.md",
    "results/TABLA_CLASIFICADOR.md": REPO / "resultados" / "clasificacion" / "TABLA_CLASIFICADOR.md",
}
CODE_COPIES = ("preprocess.py", "dinov2_plain.py")


class ReleaseRefused(RuntimeError):
    """A model or file that must not be published."""


def copy_files(root: Path, names=None) -> list[str]:
    written = []
    for name in names or COPIES:
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(COPIES[name], dest)
        written.append(name)
    return written


def check_model(model_dir: Path) -> dict:
    """Parity verdicts of one model; drops the fp16 copy that failed its rule."""
    parity_path = model_dir / "parity.json"
    if not parity_path.is_file():
        raise ReleaseRefused(f"{model_dir.name}: no parity.json; run scripts/release_parity.py")
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if not parity["pass"]:
        raise ReleaseRefused(f"{model_dir.name}: a parity gate failed ({parity['gates']})")
    fp16 = model_dir / "onnx" / "model_fp16.onnx"
    fp16_block = parity["deployment_metrics"].get("onnx_fp16", {})
    if fp16.exists() and not fp16_block.get("gate", {}).get("publish", False):
        fp16.unlink()
    return {"model": model_dir.name, "fp16_published": fp16.exists()}


def finalize(root: Path, models, limit_bytes: int = package.GITHUB_ASSET_LIMIT) -> dict:
    import build_model_card  # noqa: PLC0415

    summary = [check_model(root / m) for m in models]
    copy_files(root)
    (root / "MODEL_CARD.md").write_text(build_model_card.render(root), encoding="utf-8")
    (root / "sizes.json").unlink(missing_ok=True)
    report = package.size_report(root, limit_bytes)
    package.write_json(root / "sizes.json", {"limit_bytes": limit_bytes, "files": report})
    package.write_sha256sums(root)
    return {"models": summary, "sizes": report}


def main(argv=None) -> int:
    import build_model_card  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--release", type=Path, default=REPO / "release" / "v7")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args(argv)
    if build_model_card.main(["--check"]) != 0:
        return 1
    try:
        result = finalize(args.release, args.models)
    except (ReleaseRefused, package.AssetTooLarge) as exc:
        print(f"ABORT: {exc}")
        return 1
    for row in result["sizes"]:
        print(f"{row['mib']:>10.2f} MiB  {row['path']}")
    for model in result["models"]:
        print(model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
