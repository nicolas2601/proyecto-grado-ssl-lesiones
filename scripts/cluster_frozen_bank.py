"""Cluster-side frozen-backbone HAM10000 evaluation, stratified by skin tone.

Reproduces the frozen-backbone protocol of the Kaggle notebook
(``scripts/notebook_code_v5.py``, DATA/SPLIT/EXTRACT/PROBE/REPORT blocks) for
an unattended GPU cluster pod, and optionally joins a skin-tone manifest
(``plan-segmentacion/comun/evaluacion/fst_pseudo_manifest.py`` output) to
report the fairness gap per tone stratum. Heavy imports (torch, torchvision,
PIL, pandas, scikit-learn) are deferred into the functions that need them, so
``--help`` works with only numpy installed.

    python scripts/cluster_frozen_bank.py --ham-root /data/ham10000 \\
        --backbone ssl=dinov2_vits14:public --output-dir /data/results/bank
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dermssl import run_environment  # noqa: E402
from dermssl.data_resolver import index_images  # noqa: E402
from dermssl.feature_cache import FeatureCache, cache_key  # noqa: E402
from dermssl.label_regimes import build_regimes, stratified_regime_indices  # noqa: E402
from dermssl.metrics import UNAVAILABLE_BUCKET, stratified_by_label, summarize  # noqa: E402
from dermssl.patch_guard import assert_compatible  # noqa: E402
from dermssl.probes import headline_metric, run_probe  # noqa: E402

SPLIT_N_SPLITS = 5
SPLIT_RANDOM_STATE = 42  # matches notebook_code_v5.py SPLIT_CODE exactly
SMOKE_FRACTION = 0.08
# Notebook's min_per_class=12/6 is tuned for real HAM10000 class sizes (~80);
# a generic CLI falls back to label_regimes' own default instead.
SMOKE_MIN_PER_CLASS = 2
BACKBONE_SOURCE_PUBLIC = "public"
DEFAULT_PROBES = ("logreg_balanced", "knn")
DEFAULT_FRACTIONS = (1.0, 0.1)
DEFAULT_SEEDS = (0,)
DEFAULT_STRATA_COLUMN = "tone_tercile"
DEFAULT_MIN_N = 30
DEFAULT_RESOLUTION = 224
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Columns written by fst_pseudo_manifest.py's CSV_COLUMNS.
MANIFEST_COLUMNS = ("image_id", "fst_bucket", "tone_tercile", "status",
                     "L_median", "b_median", "ita", "n_skin_px")
STRATA_PREFERRED_PROBE = "logreg_balanced"
STRATA_PREFERRED_FRACTION = 1.0
STRATA_HEADLINE_METRIC = "macro_f1"

_SIMPLE_OPTIONS = (  # (flag, kwargs) pairs, so build_arg_parser stays one screen
    ("--strata-column", dict(default=DEFAULT_STRATA_COLUMN, help="Manifest stratification column.")),
    ("--probes", dict(nargs="+", default=list(DEFAULT_PROBES), help="dermssl.probes.run_probe names.")),
    ("--fractions", dict(nargs="+", type=float, default=list(DEFAULT_FRACTIONS), help="Label fractions.")),
    ("--seeds", dict(nargs="+", type=int, default=list(DEFAULT_SEEDS), help="Random seeds.")),
    ("--min-n", dict(type=int, default=DEFAULT_MIN_N, help="Min stratum n to report, else SKIPPED.")),
    ("--resolution", dict(type=int, default=DEFAULT_RESOLUTION, help="Input resolution in pixels.")),
    ("--batch-size", dict(type=int, default=DEFAULT_BATCH_SIZE, help="Extraction batch size.")),
    ("--num-workers", dict(type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader worker count.")),
    ("--device", dict(default=None, help="Torch device; default 'cuda' if available else 'cpu'.")),
    ("--cache-dir", dict(type=Path, default=None, help="Optional dermssl.feature_cache directory.")),
    ("--wandb-project", dict(default=None, help="Optional W&B project; never aborts if unavailable.")),
)

def parse_backbone_spec(raw: str) -> tuple[str, str, str]:
    """Parse ``NAME=ARCH:SOURCE`` into ``(name, arch, source)``."""
    name, eq, rest = raw.partition("=")
    if not eq or not name or not rest:
        raise argparse.ArgumentTypeError(f"malformed --backbone {raw!r}; expected NAME=ARCH:SOURCE")
    arch, colon, source = rest.partition(":")
    if not colon or not arch or not source:
        raise argparse.ArgumentTypeError(f"malformed --backbone {raw!r}; expected NAME=ARCH:SOURCE")
    return name, arch, source

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen-backbone HAM10000 evaluation for a GPU cluster pod, "
                     "optionally stratified by a skin-tone manifest column.",
    )
    parser.add_argument("--ham-root", required=True, type=Path,
                         help="Directory with HAM10000_metadata.csv and the *.jpg images.")
    parser.add_argument("--backbone", dest="backbones", action="append", required=True,
                         type=parse_backbone_spec, metavar="NAME=ARCH:SOURCE",
                         help="Repeatable. ARCH is a torch.hub dinov2 name; SOURCE is "
                              "'public' or a state_dict .pt path.")
    parser.add_argument("--tone-manifest", type=Path, default=None,
                         help="fst_pseudo_manifest.csv; without it, no strata are computed.")
    for flag, kwargs in _SIMPLE_OPTIONS:
        parser.add_argument(flag, **kwargs)
    parser.add_argument("--output-dir", required=True, type=Path,
                         help="Directory for bank.json, bank_rows.csv, strata.json, bank.md.")
    parser.add_argument("--smoke", action="store_true",
                         help="Small stratified subset for a fast pipeline check.")
    return parser

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)

def read_ham_metadata_csv(ham_root: Path):
    """Load HAM10000_metadata.csv from ``ham_root`` (direct child, else search)."""
    import pandas as pd
    direct = ham_root / "HAM10000_metadata.csv"
    if direct.is_file():
        return pd.read_csv(direct)
    matches = list(Path(ham_root).rglob("HAM10000_metadata.csv"))
    if not matches:
        raise FileNotFoundError(f"HAM10000_metadata.csv not found under {ham_root}")
    return pd.read_csv(matches[0])

def filter_metadata_to_available(meta, image_index: dict[str, Path]):
    """Only rows whose image is actually present on disk."""
    return meta[meta.image_id.isin(image_index)].reset_index(drop=True)

def build_split(meta) -> dict:
    """Exactly notebook_code_v5.py's SPLIT_CODE: StratifiedGroupKFold(5, shuffle, rs=42)."""
    from sklearn.model_selection import StratifiedGroupKFold
    classes = sorted(meta.dx.unique())
    cls2idx = {c: i for i, c in enumerate(classes)}
    all_ids = meta.image_id.tolist()
    y_all = np.array([cls2idx[d] for d in meta.dx])
    groups = meta.lesion_id.values
    sgkf = StratifiedGroupKFold(n_splits=SPLIT_N_SPLITS, shuffle=True,
                                 random_state=SPLIT_RANDOM_STATE)
    tr_idx, va_idx = next(sgkf.split(all_ids, y_all, groups))
    train_ids = [all_ids[i] for i in tr_idx]
    val_ids = [all_ids[i] for i in va_idx]
    lookup = meta.set_index("image_id")
    overlap = set(lookup.loc[train_ids, "lesion_id"]) & set(lookup.loc[val_ids, "lesion_id"])
    assert not overlap, f"LEAKAGE: {len(overlap)} lesions in both train and val"
    id2label = {row.image_id: cls2idx[row.dx] for row in meta.itertuples()}
    return {
        "train_ids": train_ids, "val_ids": val_ids, "classes": classes,
        "cls2idx": cls2idx, "id2label": id2label, "overlap_lesions": len(overlap),
    }

def smoke_subset_ids(ids: list[str], y: np.ndarray) -> list[str]:
    """Small stratified subset, like the notebook's SPLIT_CODE smoke block."""
    idx = stratified_regime_indices(y, SMOKE_FRACTION, seed=SPLIT_RANDOM_STATE,
                                     min_per_class=SMOKE_MIN_PER_CLASS)
    return [ids[i] for i in idx]

def load_tone_manifest_rows(path: Path) -> dict[str, dict]:
    """image_id -> row dict, straight from the fst_pseudo_manifest.py CSV."""
    rows: dict[str, dict] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[row["image_id"]] = row
    return rows

def strata_labels_for(ids: list[str], manifest_rows: dict[str, dict] | None, column: str) -> np.ndarray:
    """String label per id; missing image_id or empty/absent value -> unavailable."""
    labels = []
    for image_id in ids:
        row = manifest_rows.get(image_id) if manifest_rows else None
        value = row.get(column) if row else None
        empty = value is None or str(value).strip() == ""
        labels.append(UNAVAILABLE_BUCKET if empty else str(value))
    return np.array(labels, dtype=object)

def eval_transform(resolution: int, interpolation=None):
    """Deterministic eval transform: no augmentation, matches EXTRACT_CODE."""
    from torchvision import transforms
    interp = interpolation or transforms.InterpolationMode.BILINEAR
    return transforms.Compose([
        transforms.Resize(int(resolution * 1.14), interpolation=interp),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(list(IMAGENET_MEAN), list(IMAGENET_STD)),
    ])

def load_rgb_or_placeholder(path: Path, resolution: int):
    """Unreadable image -> black placeholder and a warning, never a crash."""
    from PIL import Image
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - any decode failure is recoverable
        print(f"[WARN] unreadable image {path}: {exc} (using black placeholder)")
        return Image.new("RGB", (resolution, resolution), (0, 0, 0))

def _make_id_dataset(ids, image_index, labels_by_id, transform, resolution):
    from torch.utils.data import Dataset
    class _IdDataset(Dataset):
        def __len__(self) -> int:
            return len(ids)
        def __getitem__(self, i: int):
            image_id = ids[i]
            image = load_rgb_or_placeholder(image_index[image_id], resolution)
            return transform(image), labels_by_id[image_id]
    return _IdDataset()

def extract_features(
    model, ids: list[str], image_index: dict[str, Path], labels_by_id: dict[str, int],
    resolution: int, device, batch_size: int, num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run every id through ``model``; no grad, autocast on cuda."""
    import torch
    from torch.utils.data import DataLoader
    transform = eval_transform(resolution)
    dataset = _make_id_dataset(ids, image_index, labels_by_id, transform, resolution)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    out = model(images)
            else:
                out = model(images)
            feats.append(out.float().cpu().numpy())
            labels.append(np.asarray(batch_labels))
    return np.concatenate(feats), np.concatenate(labels)

def construct_backbone(arch: str, source: str, resolution: int, device):
    """Build the raw (not yet frozen/eval) backbone. Monkeypatch seam for tests."""
    import torch
    if source == BACKBONE_SOURCE_PUBLIC:
        return torch.hub.load("facebookresearch/dinov2", arch, pretrained=True)
    model = torch.hub.load("facebookresearch/dinov2", arch, pretrained=False)
    state = torch.load(source, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model

def build_frozen_backbone(arch: str, source: str, resolution: int, device) -> dict:
    """Freeze, eval, move to device, and verify zero trainable parameters."""
    import torch
    model = construct_backbone(arch, source, resolution, device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval().to(device)
    trainable = sum(int(p.requires_grad) for p in model.parameters())
    assert trainable == 0, f"{arch}: backbone is not frozen ({trainable} trainable params)"
    with torch.no_grad():
        probe_input = torch.zeros(1, 3, resolution, resolution, device=device)
        dim = model(probe_input).shape[-1]
    return {"model": model, "dim": int(dim), "trainable_params": trainable}

def get_or_extract(
    cache: FeatureCache | None, backbone_name: str, split_name: str,
    resolution: int, ids: list[str], extract_fn, meta: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if cache is None:
        return extract_fn()
    key = cache_key(backbone_name, split_name, resolution, ids)
    entry = cache.get_or_extract(key, extract_fn, meta=meta)
    return entry.features, entry.labels

def run_probe_grid(
    features_by_backbone: dict[str, dict[str, tuple]],
    probes: list[str], fractions: list[float], seeds: list[int], n_classes: int,
) -> tuple[list[dict], dict]:
    """backbone x seed x fraction x probe grid. One failing probe never aborts it."""
    rows: list[dict] = []
    proba_store: dict[tuple, np.ndarray] = {}
    for backbone, split_data in features_by_backbone.items():
        x_tr, y_tr = split_data["train"]
        x_va, y_va = split_data["val"]
        for seed in seeds:
            regimes = build_regimes(y_tr, tuple(fractions), seed=seed, min_per_class=2)
            for fraction in fractions:
                idx = regimes[fraction]
                for probe in probes:
                    row, proba = _run_one_probe(
                        backbone, probe, fraction, seed, x_tr, y_tr, idx, x_va, y_va, n_classes,
                    )
                    rows.append(row)
                    if proba is not None:
                        proba_store[(backbone, probe, fraction, seed)] = proba
    return rows, proba_store

def _run_one_probe(backbone, probe, fraction, seed, x_tr, y_tr, idx, x_va, y_va, n_classes):
    try:
        proba = run_probe(probe, x_tr[idx], y_tr[idx], x_va, n_classes)
        stats = summarize(y_va, proba, n_classes)
        error = None
    except Exception as exc:  # noqa: BLE001 - a broken probe must not abort the grid
        proba = None
        stats = {k: float("nan") for k in
                 ("macro_f1", "accuracy", "balanced_accuracy",
                  "macro_precision", "macro_recall", "auc", "ece")}
        error = f"{type(exc).__name__}: {exc}"
    row = {
        "backbone": backbone, "probe": probe, "fraction": fraction, "seed": seed,
        "n_labels": int(len(idx)), "headline_metric": headline_metric(probe),
        "error": error, **stats,
    }
    return row, proba

def select_strata_probe_fraction(probes: list[str], fractions: list[float]) -> tuple[str, float]:
    if STRATA_PREFERRED_PROBE in probes and STRATA_PREFERRED_FRACTION in fractions:
        return STRATA_PREFERRED_PROBE, STRATA_PREFERRED_FRACTION
    return probes[0], fractions[0]

def compute_gap(strata: dict, metric: str) -> tuple[float | None, str | None]:
    """max-min of ``metric`` across OK, non-unavailable labels; None if <2 qualify."""
    qualifying = {
        label: entry[metric]
        for label, entry in strata.items()
        if entry.get("status") == "OK" and label != UNAVAILABLE_BUCKET
    }
    if len(qualifying) < 2:
        return None, None
    worst_label = min(qualifying, key=qualifying.get)
    gap = max(qualifying.values()) - min(qualifying.values())
    return gap, worst_label

def gap_is_established(strata: dict, metric: str) -> bool | None:
    """Whether the best and worst strata are actually separated.

    A gap between point estimates says nothing on its own: the first bank
    reported the light tercile as the worst in every backbone, but that
    stratum had 183 rows against 957 and the ordering was noise. True when
    the extreme strata's intervals do not overlap, False when they do, and
    None when there are no intervals to judge - which is not the same as
    "no gap" and must not be reported as one.
    """
    qualifying = {
        label: entry for label, entry in strata.items()
        if entry.get("status") == "OK" and label != UNAVAILABLE_BUCKET
    }
    if len(qualifying) < 2:
        return None
    low_key, high_key = f"{metric}_ci95_low", f"{metric}_ci95_high"
    if any(low_key not in entry or high_key not in entry for entry in qualifying.values()):
        return None
    best = max(qualifying, key=lambda label: qualifying[label][metric])
    worst = min(qualifying, key=lambda label: qualifying[label][metric])
    if best == worst:
        return False
    return qualifying[worst][high_key] < qualifying[best][low_key]


def build_strata_entry(
    y_val: np.ndarray, proba: np.ndarray, labels: np.ndarray, column: str,
    probe: str, fraction: float, n_classes: int, min_n: int,
) -> dict:
    strat = stratified_by_label(y_val, proba, labels, n_classes, min_n=min_n)
    gap_macro_f1, worst_label = compute_gap(strat, STRATA_HEADLINE_METRIC)
    gap_auc, _ = compute_gap(strat, "auc")
    return {
        "column": column, "probe": probe, "fraction": fraction, "labels": strat,
        "gap_macro_f1": gap_macro_f1, "gap_auc": gap_auc, "worst_label": worst_label,
        "gap_established": gap_is_established(strat, STRATA_HEADLINE_METRIC),
    }

def json_safe(obj):
    """Recursively turn NaN into null so bank.json never contains ``NaN``."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return None if value != value else value
    if isinstance(obj, np.integer):
        return int(obj)
    return obj

def build_bank_payload(mode: str, backbone_meta: dict, split_meta: dict,
                        rows: list[dict], strata: dict) -> dict:
    return {"mode": mode, "backbones": backbone_meta, "split": split_meta,
            "rows": rows, "strata": strata}

def _write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def _probe_table(rows: list[dict], probe: str) -> list[str]:
    headline = headline_metric(probe)
    ok_rows = [r for r in rows if r["probe"] == probe and r["error"] is None]
    lines = [f"## Probe: {probe}", "", "| backbone | fraction | headline_metric | mean |",
              "|---|---|---|---|"]
    for backbone in sorted({r["backbone"] for r in ok_rows}):
        for fraction in sorted({r["fraction"] for r in ok_rows}):
            values = [r[headline] for r in ok_rows
                      if r["backbone"] == backbone and r["fraction"] == fraction]
            if values:
                lines.append(f"| {backbone} | {fraction:.0%} | {headline} | "
                             f"{sum(values) / len(values):.4f} |")
    return [*lines, ""]

def _strata_table(backbone: str, entry: dict) -> list[str]:
    lines = [f"## Strata: {backbone} ({entry['column']}, probe={entry['probe']}, "
             f"fraction={entry['fraction']:.0%})",
             "", "| label | n | macro_f1 | auc |", "|---|---|---|---|"]
    for label, stats in entry["labels"].items():
        if stats["status"] != "OK":
            lines.append(f"| {label} | {stats['count']} | SKIPPED | SKIPPED |")
        else:
            lines.append(f"| {label} | {stats['count']} | {stats['macro_f1']:.4f} | "
                         f"{stats['auc']:.4f} |")
    return [*lines, ""]

def render_markdown(bank: dict) -> str:
    lines = [f"# Frozen-backbone bank ({bank['mode']})", ""]
    for probe in sorted({r["probe"] for r in bank["rows"]}):
        lines += _probe_table(bank["rows"], probe)
    for backbone, entry in bank["strata"].items():
        lines += _strata_table(backbone, entry)
    return "\n".join(lines) + "\n"

def write_outputs(output_dir: Path, bank: dict, rows: list[dict], environment=None) -> None:
    """``environment`` goes in bank.json; strata.json stays a map of backbones."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_bank = json_safe({**bank, "environment": environment or run_environment.snapshot()})
    (output_dir / "bank.json").write_text(json.dumps(safe_bank, indent=2) + "\n", encoding="utf-8")
    (output_dir / "strata.json").write_text(
        json.dumps(json_safe(bank["strata"]), indent=2) + "\n", encoding="utf-8")
    _write_rows_csv(output_dir / "bank_rows.csv", json_safe(rows))
    (output_dir / "bank.md").write_text(render_markdown(bank), encoding="utf-8")

def maybe_log_wandb(args: argparse.Namespace, rows: list[dict], strata: dict) -> None:
    if not args.wandb_project:
        return
    try:
        import wandb
        run = wandb.init(project=args.wandb_project)
        if rows:
            columns = list(rows[0].keys())
            table = wandb.Table(columns=columns)
            for row in rows:
                table.add_data(*[row.get(c) for c in columns])
            wandb.log({"bank_rows": table})
        for name, entry in strata.items():
            run.summary[f"{name}/gap_macro_f1"] = entry.get("gap_macro_f1")
            run.summary[f"{name}/gap_auc"] = entry.get("gap_auc")
        wandb.finish()
    except Exception as exc:  # noqa: BLE001 - logging must never abort the run
        print(f"[WARN] wandb logging skipped: {exc}")

def _extract_all_backbones(args, image_index, split, train_ids, val_ids, device, cache):
    backbone_meta: dict[str, dict] = {}
    features_by_backbone: dict[str, dict] = {}
    for name, arch, source in args.backbones:
        resolution = assert_compatible(args.resolution, arch)
        built = build_frozen_backbone(arch, source, resolution, device)
        model = built["model"]
        def extract(ids, model=model, resolution=resolution):
            return extract_features(model, ids, image_index, split["id2label"], resolution,
                                     device, args.batch_size, args.num_workers)
        x_tr, y_tr = get_or_extract(cache, name, "train", resolution, train_ids,
                                     lambda: extract(train_ids))
        x_va, y_va = get_or_extract(cache, name, "val", resolution, val_ids,
                                     lambda: extract(val_ids))
        backbone_meta[name] = {"arch": arch, "source": source, "dim": built["dim"],
                                "resolution": resolution,
                                "frozen_trainable_params": built["trainable_params"]}
        features_by_backbone[name] = {"train": (x_tr, y_tr), "val": (x_va, y_va)}
        del model
    return backbone_meta, features_by_backbone

def _build_strata(args, features_by_backbone, val_ids, proba_store, n_classes) -> dict:
    if not args.tone_manifest:
        return {}
    manifest_rows = load_tone_manifest_rows(Path(args.tone_manifest))
    labels = strata_labels_for(val_ids, manifest_rows, args.strata_column)
    probe, fraction = select_strata_probe_fraction(args.probes, args.fractions)
    seed = args.seeds[0]
    strata = {}
    for name, split_data in features_by_backbone.items():
        key = (name, probe, fraction, seed)
        if key not in proba_store:
            continue
        _, y_va = split_data["val"]
        strata[name] = build_strata_entry(
            y_va, proba_store[key], labels, args.strata_column, probe, fraction,
            n_classes, args.min_n,
        )
    return strata

def _resolve_device(requested: str | None):
    import torch
    name = requested or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def _load_split(args):
    meta_raw = read_ham_metadata_csv(Path(args.ham_root))
    image_index = index_images([Path(args.ham_root)])
    meta = filter_metadata_to_available(meta_raw, image_index)
    split = build_split(meta)
    train_ids, val_ids = split["train_ids"], split["val_ids"]
    if args.smoke:
        y_train_full = np.array([split["id2label"][i] for i in train_ids])
        y_val_full = np.array([split["id2label"][i] for i in val_ids])
        train_ids = smoke_subset_ids(train_ids, y_train_full)
        val_ids = smoke_subset_ids(val_ids, y_val_full)
    return image_index, split, train_ids, val_ids

def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    image_index, split, train_ids, val_ids = _load_split(args)
    n_classes = len(split["classes"])
    device = _resolve_device(args.device)
    cache = FeatureCache(Path(args.cache_dir)) if args.cache_dir else None
    backbone_meta, features_by_backbone = _extract_all_backbones(
        args, image_index, split, train_ids, val_ids, device, cache,
    )
    rows, proba_store = run_probe_grid(
        features_by_backbone, args.probes, args.fractions, args.seeds, n_classes,
    )
    strata = _build_strata(args, features_by_backbone, val_ids, proba_store, n_classes)
    bank = build_bank_payload(
        mode="smoke" if args.smoke else "full",
        backbone_meta=backbone_meta,
        split_meta={"train": len(train_ids), "val": len(val_ids), "n_classes": n_classes,
                    "classes": split["classes"], "overlap_lesions": split["overlap_lesions"]},
        rows=rows, strata=strata,
    )
    write_outputs(Path(args.output_dir), bank, rows, environment=run_environment.snapshot(args))
    maybe_log_wandb(args, rows, strata)
    return bank


if __name__ == "__main__":
    main()