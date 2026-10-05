"""Rebuild per-image val probabilities of the ISIC 2020 fine-tuned models (inference only).

Reads the best-epoch checkpoint each run saved, runs the SAME evaluation pass as
``torch_run.run`` and writes ``<out-dir>/<model>.probas.npz`` (keys ``y``, ``proba``,
``val_ids``, ``groups`` like ``isic_eval``, plus ``checkpoint_sha256`` and ``model``) and a sidecar
``ft_isic2020_probas.sidecar.json``. A model whose recomputed metrics differ from its results JSON
by more than 1e-4 (or whose val ids differ from the reference ``B71__clean.probas.npz``) is recorded
as "no pareado" and writes no npz; the other models still run.

    python scripts/ft_val_probas.py --data-root ~/work/datos \
        --ckpt-root ~/work/proyectonp/salidas/bench/ft_isic2020
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dermssl.finetune import protocol  # noqa: E402
from dermssl.finetune.config import ConfigError, FinetuneConfig, validate  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REFERENCE = ("resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/"
             "B71__clean.probas.npz")
OUT_DIR = "resultados/banco_completo/ft_isic2020/probas"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=protocol.MODELS, default=list(protocol.MODELS))
    p.add_argument("--data-root", required=True, help="folder with isic2020/")
    p.add_argument("--ckpt-root", required=True, help="<root>/<model>/ft_<model>_isic2020.pth")
    p.add_argument("--json-root", default="", help="<root>/<model>/results_<model>_ft_isic2020.json "
                   "(default: --ckpt-root)")
    p.add_argument("--out-dir", default=str(REPO / OUT_DIR))
    p.add_argument("--reference", default=str(REPO / REFERENCE), help="B71__clean.probas.npz")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--batch", type=int, default=protocol.BATCH)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from dermssl.finetune import val_probas  # imports torch

    def make_cfg(model: str) -> FinetuneConfig:
        return validate(FinetuneConfig(
            model=model, dataset="isic2020", data_root=args.data_root,
            output_dir=args.out_dir, num_workers=args.num_workers, batch_size=args.batch))

    try:
        make_cfg(args.models[0])
    except ConfigError as exc:
        sys.exit(f"config error: {exc}")
    reference = val_probas.load_reference_ids(Path(args.reference))
    sidecar = val_probas.pair_models(
        args.models, make_cfg, Path(args.ckpt_root), Path(args.json_root or args.ckpt_root),
        Path(args.out_dir), reference)
    for rec in sidecar["models"]:
        print(f"{rec['model']:<16} {rec['status']}  {rec.get('reason', '')}")
    return 0 if all(r["status"] == "pareado" for r in sidecar["models"]) else 1


if __name__ == "__main__":
    sys.exit(main())
