r"""E1-ISIC (docs/33): the fixed v7.1 configs on the EXISTING ISIC 2019 / ISIC 2020 splits.

Configs (nothing is selected on val; the probe is fitted on the dataset TRAIN split):

* ``B71``    = ``B_s42_a1.0@224|cls4|balanced|tta`` (alpha 1.0 = the seed-42 final itself)
* ``A71``    = ``A_soup_a1.0@224|cls_attn|la0.5|tta`` (uniform soup of the A s42 and A s43 finals)
* ``PUBLIC`` = public DINOv2 ViT-B/14 under the B71 protocol and under the A71 protocol (two cells)

``--pool-excludes-val`` states which weights are used: given, the finals are the clean ones
(``<weights-root>/final_{A,B}_s{42,43}``, pool without the val ids; ``transductivo: false``); absent, the
original v7 finals (``final_A``, ``final_A_s43``, ``final_B``; the SSL pool contained the val images;
``transductivo: true``). The label is checked against each ``run.json`` ``exclude_ids_sha256`` and the run
aborts on a mismatch (``--allow-unverified-pool`` only when the field is absent).

Before anything heavy the rebuilt split is compared with the committed one (val supports and val sha256) and
the run REFUSES to continue if they differ. ``StratifiedGroupKFold`` (ISIC 2020) only reproduces with
scikit-learn <= 1.7.2: put the isolated 1.7.2 in front (``PYTHONPATH=$HOME/work/skl_try/1.7.2``).

Per cell: ``<out-dir>/<dataset>/<cell>.json`` (+ ``.probas.npz``, ``.environment.json``); paired bootstrap
(1000, seed 20260923; by ``patient_id`` for ISIC 2020, by image for ISIC 2019) against public and the other
config go to ``<out-dir>/<dataset>/contrasts/`` as soon as both cells exist. Relaunching the same line skips
finished cells; a failing cell is retried twice and listed at the end. ``--limit N`` writes to
``<dataset>_smoke_<N>`` and needs ``--allow-dirty`` on an uncommitted tree.

Pod (P=$HOME/work/proyectonp; clean clone; TORCH_HOME with the dinov2 hub repo)::

    export TORCH_HOME=$HOME/work/.torch OMP_NUM_THREADS=4 PYTHONPATH=$HOME/work/skl_try/1.7.2
    D=$HOME/work/datos; O=$P/salidas/bench/isic_eval
    I19="--dataset isic2019 --data-root $D/isic2019_fullres --image-dir $D/isic2019/train-image"
    I20="--dataset isic2020 --data-root $D/isic2020"
    python scripts/isic_eval_run.py $I19 --dry-run            # split check only, no GPU
    for DS in "$I19" "$I20"; do
      python scripts/isic_eval_run.py $DS --config PUBLIC --weights-root $P/salidas/v7 --out-dir $O
      for C in B71 A71; do   # clean, then original (transductive)
        python scripts/isic_eval_run.py $DS --config $C --weights-root $P/salidas/bench/v7clean \
          --pool-excludes-val --out-dir $O
        python scripts/isic_eval_run.py $DS --config $C --weights-root $P/salidas/v7 --out-dir $O
      done
    done
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import v71_io as io  # noqa: E402

from dermssl import e3_store, probe_threads, run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl.isic_eval import protocol as pr  # noqa: E402
from dermssl.isic_eval import runner as rn  # noqa: E402
from dermssl.isic_eval import scoring  # noqa: E402
from dermssl.isic_eval import splits as sp  # noqa: E402
from dermssl.isic_eval import weights as wt  # noqa: E402

PUBLIC_LABEL = "torch.hub facebookresearch/dinov2 dinov2_vitb14 pretrained"
NEEDED_FINALS = {"B": ("s42",), "A": ("s42", "s43")}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", choices=sorted(sp.LOADERS), required=True)
    p.add_argument("--data-root", type=Path, required=True, help="folder holding the ground-truth / metadata CSV")
    p.add_argument("--image-dir", type=Path, default=None, help="folder searched for <id>.jpg (default: data-root)")
    p.add_argument("--config", choices=sorted(pr.CLI_CONFIGS), default=None)
    p.add_argument("--weights-root", type=Path, default=None)
    p.add_argument("--pool-excludes-val", action="store_true",
                   help="weights are the clean finals (SSL pool without the val ids): transductivo=false")
    p.add_argument("--allow-unverified-pool", action="store_true", help="run.json lacks exclude_ids_sha256")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--decode-workers", type=int, default=8)
    p.add_argument("--ci-resamples", type=int, default=scoring.N_RESAMPLES)
    p.add_argument("--limit", type=int, default=None, help="smoke: N train and N val images")
    p.add_argument("--keep-features", action="store_true")
    p.add_argument("--allow-dirty", action="store_true", help="smoke only")
    p.add_argument("--dry-run", action="store_true", help="rebuild and verify the split, print the plan, stop")
    args = p.parse_args(argv)
    if not args.dry_run and (args.config is None or args.weights_root is None or args.out_dir is None):
        p.error("--config, --weights-root and --out-dir are required unless --dry-run")
    return args


class LayoutBank(io.WeightBank):
    """``WeightBank`` over the original or the clean folder layout."""

    def __init__(self, root: Path, pool_excludes_val: bool):
        super().__init__(root)
        self.clean = pool_excludes_val

    def final_path(self, arm: str, seed: str) -> Path:
        return self.weights_dir / wt.final_dir_name(arm, int(seed[1:]), self.clean) / io.WEIGHTS_FILE


def needed_finals(configs) -> list[tuple[str, str]]:
    arms = {pr.parse(c)["weights"][0] for c in configs if not pr.is_public(c)}
    return [(arm, seed) for arm in sorted(arms) for seed in NEEDED_FINALS[arm]]


def verify_pool(bank: LayoutBank, configs, args) -> dict:
    finals = needed_finals(configs)
    if not finals:
        return {}
    manifests = {}
    for arm, seed in finals:
        run_json = bank.final_path(arm, seed).parent / "run.json"
        manifests[f"{arm}_{seed}"] = io.read_json(run_json) if run_json.is_file() else {}
    try:
        return wt.check_pool_label(manifests, args.pool_excludes_val, args.allow_unverified_pool)
    except wt.PoolLabelError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc


def weights_digest(bank: LayoutBank, config: str) -> str:
    """sha256 over the final files behind a config (public: the hub identity, no local file)."""
    name = pr.parse(config)["weights"]
    if name == "public":
        return PUBLIC_LABEL
    arm = name[0]
    return "+".join(rp.file_sha256(bank.final_path(arm, s)) for s in NEEDED_FINALS[arm])


def build_extractor(args, ds, bank: LayoutBank):
    import torch  # noqa: PLC0415

    from dermssl import torch_v71 as tv  # noqa: PLC0415

    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    store = io.ImageStore(args.out_dir / "inputs", ds.paths, args.decode_workers)

    def extract(weights, split, ids, n_views, parts):
        images = store.array(ids, pr.RESOLUTION, f"{ds.name}_{split}")
        model = bank.model(bank.state(weights))
        got = tv.extract_split(model, images, device=device, batch_size=args.batch_size, n_views=n_views)
        return {p: got[p] for p in parts}

    return extract


def load_dataset(args) -> sp.EvalDataset:
    loader, csv_name = sp.LOADERS[args.dataset]
    csv = next(args.data_root.expanduser().rglob(csv_name), None)
    if csv is None:
        raise SystemExit(f"ABORT: no {csv_name} under {args.data_root}")
    image_dir = (args.image_dir or args.data_root).expanduser()
    try:
        return loader(csv, image_dir)
    except sp.SplitMismatchError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc


def revision(args) -> str:
    try:
        return rp.clean_revision(REPO)
    except rp.DirtyTreeError as exc:
        if args.allow_dirty and args.limit:
            return "dirty"
        raise SystemExit(f"ABORT: {exc}") from exc


def stamp_sidecars(store, names, args) -> None:
    for name in names:
        path = store.path(name)
        if path.is_file() and not run_environment.sidecar_path(path).is_file():
            run_environment.write_sidecar(path, args, loky_inner_threads=probe_threads.PROBE_BLAS_THREADS)


def run_configs(args, ds, configs) -> int:
    dataset_dir = args.out_dir.expanduser() / (f"{ds.name}_smoke_{args.limit}" if args.limit else ds.name)
    store, contrasts = e3_store.CellStore(dataset_dir), e3_store.CellStore(dataset_dir / "contrasts")
    bank = LayoutBank(args.weights_root.expanduser(), args.pool_excludes_val)
    pool = verify_pool(bank, configs, args)
    run_ds = sp.limited(ds, args.limit)
    extract = build_extractor(args, run_ds, bank)
    provenance = {"code_revision": revision(args), "pool_check": pool}
    for config in configs:
        status = rn.run_cell(store, run_ds, config, extract, pool_excludes_val=args.pool_excludes_val,
                             weights_sha=weights_digest(bank, config), limit=args.limit,
                             n_resamples=args.ci_resamples, keep_features=args.keep_features,
                             provenance=provenance)
        print(f"[{pr.cell_name(config, args.pool_excludes_val)}] {status}")
        stamp_sidecars(store, [pr.cell_name(config, args.pool_excludes_val)], args)
    done = rn.run_all_contrasts(store, contrasts, n_resamples=args.ci_resamples)
    stamp_sidecars(contrasts, done, args)
    failures = store.failures + contrasts.failures
    for name, error in failures:
        print(f"FAILED {name}: {error}")
    return 1 if failures else 0


def main(argv=None) -> int:
    args = parse_args(argv)
    ds = load_dataset(args)
    print(f"{ds.name}: split verified (sklearn {ds.identity['sklearn_version']}), n_train {ds.identity['n_train']}, "
          f"n_val {ds.identity['n_val']}, val supports {ds.identity['val_supports']}, "
          f"val sha256 {ds.identity['val_ids_sha256'][:12]}")
    if args.dry_run:
        return 0
    return run_configs(args, ds, list(pr.CLI_CONFIGS[args.config]))


if __name__ == "__main__":
    sys.exit(main())
