"""PAD-UFES-20 descriptive addendum of docs/30 7: A71 and seed 44. Single run, on the pod.

Two model families, each on the protocol it was frozen with, both fitted on
the whole C1 fold-0 train of HAM10000 and applied to PAD without seeing it:

* A71, the v7.1 winner for A (``selection.json``, pinned by sha256), and public
  on the same protocol: soup s42+s43, ``cls_attn``, ``la0.5``, D4 TTA, by the
  v7.1 val-read code (``v71_io`` weights and inputs, ``torch_v71.extract_split``,
  ``v71_grid.unit_probas``). The train inputs are v7.1's cached array.
* A* and B* of seed 44, and public, on the docs/30 2 protocol (``logreg`` on
  CLS at 224), as ``pad_ufes_eval`` reads s42/s43.

Readings are ``dermssl.pad_ufes_addendum``: the first pass's, without any
verdict. Every probe fit runs under ``probe_threads.PROBE_BLAS_THREADS``.

Before touching any data it refuses unless the docs/30 7 block was committed
before this script and is unchanged, the tree is clean, ``--single-evaluation``
is given, the output does not exist, the selection matches its sha256, the
v7.1 train inputs are cached, and (later) every PAD image matches its manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import c1_final_eval as c1  # noqa: E402
import c1_select_variant as sel  # noqa: E402
import cluster_frozen_bank as bank  # noqa: E402
import pad_ufes_eval as ev  # noqa: E402
import v71_io as io  # noqa: E402
import v71_val  # noqa: E402

from dermssl import feature_cache as fc  # noqa: E402
from dermssl import pad_ufes_addendum as pa  # noqa: E402
from dermssl import pad_ufes_files as pf  # noqa: E402
from dermssl import pad_ufes_guard as pg  # noqa: E402
from dermssl import pad_ufes_readings as pr  # noqa: E402
from dermssl import probe_threads, probes  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl import t4_sog  # noqa: E402
from dermssl import v71_grid as vg  # noqa: E402
from dermssl import v71_plan as plan  # noqa: E402
from dermssl import v71_protocol as vp  # noqa: E402

#: The commit that registered docs/30 7 on its own, before any addendum code.
REGISTRATION_COMMIT = "76aa91e7b68cbfd55bd93aa857f6587efd5717b7"
BLOCK = ("<!-- registro-pad-adenda-a71-s44:inicio -->", "<!-- registro-pad-adenda-a71-s44:fin -->")
THIS_SCRIPT = "scripts/pad_ufes_addendum.py"
SELECTION = REPO / "resultados" / "clasificacion" / "v71" / "selection.json"
RESOLUTION = vp.parse_config_id(pa.A71_CONFIG)["resolution"]
TRAIN_TAG, PAD_TAG = "train", "pad"
S44_ROLES = (pr.PUBLIC, pr.role("plain", pa.SEED), pr.role("proposed", pa.SEED))
S44_PROTOCOL = "logreg on CLS @224, no TTA (docs/30 2.3, as the first pass)"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--single-evaluation", action="store_true", help="required: PAD is read once")
    for flag in ("--manifest", "--pad-root", "--ham-root", "--split-index", "--holdout", "--exclude-ids-file",
                 "--weights-dir", "--v71-inputs", "--pad-cache", "--out"):
        parser.add_argument(flag, type=Path, required=True)
    parser.add_argument("--plain-s44", required=True, help="ARCH:SOURCE of A* seed 44")
    parser.add_argument("--proposed-s44", required=True, help="ARCH:SOURCE of B* seed 44")
    parser.add_argument("--public", default="dinov2_vitb14:public")
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--probe-jobs", type=int, default=2)
    return parser.parse_args(argv)


def check_registration() -> str:
    return pg.check_registration(REPO, commit=REGISTRATION_COMMIT, doc=pg.REGISTRATION_DOC,
                                 script=THIS_SCRIPT, start=BLOCK[0], end=BLOCK[1])


def preflight(args: argparse.Namespace) -> tuple[str, str]:
    """Everything that can refuse, before any data is read."""
    if not args.single_evaluation:
        raise SystemExit("ABORT: exige --single-evaluation (PAD se lee una vez)")
    if args.out.expanduser().exists():
        raise SystemExit(f"ABORT: {args.out} ya existe; la evaluacion no se repite")
    try:
        registration = check_registration()
        return registration, rp.clean_revision(REPO)
    except (pg.RegistrationError, rp.DirtyTreeError) as exc:
        raise SystemExit(f"ABORT: {exc}") from exc


def check_selection(path: Path) -> dict:
    """``{A71: config, public_A71: config}`` from the frozen v7.1 selection, refused if it moved."""
    sha = rp.file_sha256(path)
    if sha != pa.SELECTION_SHA256:
        raise SystemExit(f"ABORT: {path} tiene sha256 {sha[:12]}, la adenda fijo {pa.SELECTION_SHA256[:12]}")
    entries = plan.val_entries(io.read_json(path))
    chosen = {pa.A71: entries["A71"], pa.PUBLIC_A71: entries["public_A71"]}
    expected = {pa.A71: pa.A71_CONFIG, pa.PUBLIC_A71: pa.PUBLIC_A71_CONFIG}
    if chosen != expected:
        raise SystemExit(f"ABORT: la seleccion nombra {chosen}, la adenda registro A71 = {expected}")
    return chosen


def train_inputs_path(inputs_dir: Path, train_ids) -> Path:
    """v7.1's decoded train array (``v71_io.ImageStore`` naming); never re-decoded here."""
    suffix = t4_sog.input_suffix(t4_sog.IDENTITY)
    path = inputs_dir / f"{TRAIN_TAG}{suffix}_{RESOLUTION}_{fc.fingerprint_ids(list(train_ids))}.npy"
    if not path.is_file():
        raise SystemExit(f"ABORT: falta {path}; A71 se ajusta sobre las entradas cacheadas de la lectura v7.1")
    return path


def check_s44_runs(runs: dict, exclude_sha: str) -> None:
    """A* and B* of seed 44: one revision, one exclude list (the finals' one), seed 44."""
    from dermssl import final_eval as fe  # noqa: PLC0415

    try:
        fe.check_runs_consistent(runs)
        fe.check_same_seed(runs)
    except ValueError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    for role, run in runs.items():
        if run.get("seed") != pa.SEED:
            raise SystemExit(f"ABORT: {role} es de la semilla {run.get('seed')}, no {pa.SEED}")
        if run.get("exclude_ids_sha256") != exclude_sha:
            raise SystemExit(f"ABORT: {role}: el run.json no uso el exclude {exclude_sha[:12]}")


def context_args(args: argparse.Namespace) -> argparse.Namespace:
    """What ``v71_io.load_context`` reads: the C1 split, its index and the docs/28 holdout."""
    return argparse.Namespace(ham_root=args.ham_root, image_dir=args.ham_root, split_index=args.split_index,
                              holdout=args.holdout, limit=None)


def torch_device(name: str | None):
    import torch  # noqa: PLC0415

    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def extract_split(model, images: np.ndarray, device, batch_size: int, tta: bool) -> dict:
    """``torch_v71.extract_split``: the identity view, or the eight D4 views for TTA."""
    from dermssl import torch_v71 as tv  # noqa: PLC0415

    return tv.extract_split(model, images, device=device, batch_size=batch_size,
                            n_views=tv.N_D4 if tta else 1)


def pad_inputs(args: argparse.Namespace, pad: dict) -> np.ndarray:
    """PAD decoded the v7.1 way (resize + centre crop, identity input), cached under ``--pad-cache``."""
    store = io.ImageStore(args.pad_cache.expanduser(), pad["paths"], args.decode_workers)
    return store.array(pad["ids"], RESOLUTION, PAD_TAG)


def s44_probas(specs: dict, split: dict, image_index: dict, pad: dict, device, args) -> dict:
    """``pad_ufes_eval.probe_probas`` for public, A* s44 and B* s44, with the fit pinned."""
    pad_labels = {i: pr.HAM_CLASSES.index(label) for i, label in zip(pad["ids"], pad["ham_labels"])}
    out = {}
    for role, spec in specs.items():
        _, arch, source = bank.parse_backbone_spec(f"x={spec}")
        model = bank.build_frozen_backbone(arch, source, sel.RESOLUTION, device)["model"]
        extract = lambda ids, index, labels: bank.extract_features(  # noqa: E731
            model, ids, index, labels, sel.RESOLUTION, device, args.batch_size, args.num_workers)
        x_fit, y_fit = extract(split["train_ids"], image_index, split["id2label"])
        x_pad, _ = extract(pad["ids"], pad["paths"], pad_labels)
        out[role] = probe_threads.call_pinned(probes.run_probe, sel.PROBE, x_fit, y_fit, x_pad,
                                              len(pr.HAM_CLASSES))
        print(f"[s44] {role} listo")
    return out


def a71_probas(entries: dict, weights, train_x: np.ndarray, y_fit: np.ndarray, pad_x: np.ndarray,
               device, args) -> dict:
    """Each v7.1 configuration as the val read scores it (``v71_val._score_entries``), on PAD."""
    out = {}
    for role, cid in entries.items():
        cfg = vp.parse_config_id(cid)
        model = weights.model(weights.state(cfg["weights"]))
        fit_parts = extract_split(model, train_x, device, args.batch_size, tta=False)
        pad_parts = extract_split(model, pad_x, device, args.batch_size, tta=cfg["tta"])
        cells = vg.unit_probas(fit_parts, y_fit, pad_parts, len(pr.HAM_CLASSES),
                               features=[cfg["feature"]], n_jobs=args.probe_jobs)
        out[role] = cells[(cfg["feature"], cfg["weighting"], cfg["tta"])]
        print(f"[a71] {role} = {cid} listo")
    return out


def check_inputs(args: argparse.Namespace) -> dict:
    """Split, finals and inputs, all checked before the first forward pass."""
    ctx = io.load_context(context_args(args))
    if tuple(ctx["classes"]) != pr.HAM_CLASSES:
        raise SystemExit(f"ABORT: clases de HAM {ctx['classes']} != {pr.HAM_CLASSES}")
    weights = io.WeightBank(args.weights_dir.expanduser())
    v71_guard = v71_val._guard_val(args, ctx, weights)  # val protocol, s42/s43 finals, exclude list
    specs = dict(zip(S44_ROLES, (args.public, args.plain_s44, args.proposed_s44)))
    runs = c1.run_manifests(specs)
    check_s44_runs(runs, v71_guard["exclude_ids_sha256"])
    train_path = train_inputs_path(args.v71_inputs.expanduser(), ctx["ids"]["train"])
    return {"ctx": ctx, "weights": weights, "v71_guard": v71_guard, "specs": specs, "runs": runs,
            "train_path": train_path}


def provenance(checked: dict, pad: dict, entries: dict, registration: str, revision: str) -> dict:
    runs = {**checked["v71_guard"]["runs"],
            **{r: {k: run.get(k) for k in ("code_revision", "exclude_ids_sha256", "seed")}
               for r, run in checked["runs"].items()}}
    return {"code_revision": revision, "registration_commit": registration,
            "registration_block": BLOCK[0], "selection_sha256": pa.SELECTION_SHA256,
            "protocols": {**{role: S44_PROTOCOL for role in S44_ROLES}, **entries},
            "specs": checked["specs"], "runs": runs,
            "exclude_ids_sha256": checked["v71_guard"]["exclude_ids_sha256"],
            "n_fit": len(checked["ctx"]["ids"]["train"]), "train_inputs": checked["train_path"].name,
            "manifest_sha256": pad["manifest_sha256"], "dataset_doi": pf.DATASET_DOI, "license": pf.LICENSE}


def write_outputs(out: Path, document: dict, args: argparse.Namespace, pad: dict) -> None:
    out.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    # Every fit (s44 logreg and the v7.1 units) ran under PROBE_BLAS_THREADS; the sidecar says so.
    run_environment.write_sidecar(out, args, loky_inner_threads=probe_threads.PROBE_BLAS_THREADS)
    print(ev.write_fst_table(out, pad["fst"]))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registration, revision = preflight(args)
    entries = check_selection(args.selection.expanduser())
    checked = check_inputs(args)
    pad = ev.load_pad(args.manifest.expanduser(), args.pad_root.expanduser())
    ctx, device = checked["ctx"], torch_device(args.device)
    probas = s44_probas(checked["specs"], ctx["split"], ctx["image_index"], pad, device, args)
    train_x = np.load(checked["train_path"], mmap_mode="r")
    probas.update(a71_probas(entries, checked["weights"], train_x, io.labels_of(ctx, ctx["ids"]["train"]),
                             pad_inputs(args, pad), device, args))
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".probas.npz"), ids=np.array(pad["ids"]), **probas)
    document = pa.build_document(probas, pad, provenance(checked, pad, entries, registration, revision))
    write_outputs(out, document, args, pad)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
