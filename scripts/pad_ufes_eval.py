"""PAD-UFES-20 zero-shot evaluation of the v7 finals (docs/30). Single run, on the pod.

The C1 ``logreg`` probe is fitted on the whole C1 fold-0 train of HAM10000
(the protocol of ``c1_final_eval.py --split val``, same extraction, same
fingerprint check) and applied, without seeing any PAD image, to the PAD
images of the manifest. Readings, strata and verdicts are
``dermssl.pad_ufes_readings``; nothing is chosen after the data.

Before touching any data it refuses to run unless the docs/30 prediction was
committed before this script (``dermssl.pad_ufes_guard``), the tree is
clean, and every image still matches its manifest sha256.

    python scripts/pad_ufes_eval.py --single-evaluation \
        --manifest $P/manifest.json --pad-root $P --ham-root ~/work/datos/ham10000 \
        --plain-s42 dinov2_vitb14:$V/final_A/simsiam_dinov2_backbone_final.pt \
        --proposed-s42 dinov2_vitb14:$V/final_B/simsiam_dinov2_backbone_final.pt \
        --plain-s43 dinov2_vitb14:$V/s43/final_A/simsiam_dinov2_backbone_final.pt \
        --proposed-s43 dinov2_vitb14:$V/s43/final_B/simsiam_dinov2_backbone_final.pt \
        --out resultados/clasificacion/pad_ufes/pad_ufes_eval.json
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
from c1_build_val_index import rebuild_split  # noqa: E402
from pad_ufes_manifest import MANIFEST_VERSION  # noqa: E402

from dermssl import final_eval as fe  # noqa: E402
from dermssl import pad_ufes as pu  # noqa: E402
from dermssl import pad_ufes_files as pf  # noqa: E402
from dermssl import pad_ufes_guard as pg  # noqa: E402
from dermssl import pad_ufes_readings as pr  # noqa: E402
from dermssl import probes  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--single-evaluation", action="store_true", help="required: PAD is read once")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pad-root", type=Path, required=True)
    parser.add_argument("--ham-root", type=Path, required=True)
    for role in pr.roles():
        if role != pr.PUBLIC:
            parser.add_argument(f"--{role.replace('_', '-')}", required=True, help="ARCH:SOURCE")
    parser.add_argument("--public", default="dinov2_vitb14:public")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def load_pad(manifest_path: Path, pad_root: Path) -> dict:
    """The manifest's images, each re-hashed: a file that changed since the manifest aborts."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise pf.IntegrityError(f"manifest_version {manifest.get('manifest_version')} != {MANIFEST_VERSION}")
    rows = manifest["images"]
    if len(rows) != pu.EXPECTED_IMAGES:
        raise pf.IntegrityError(f"el manifiesto tiene {len(rows)} imagenes, PAD publica {pu.EXPECTED_IMAGES}")
    paths = {}
    for row in rows:
        path = pad_root / row["path"]
        if pf.sha256_of(path) != row["sha256"]:
            raise pf.IntegrityError(f"{row['img_id']}: el archivo cambio desde el manifiesto")
        paths[row["img_id"]] = path
    column = lambda key: np.array([row[key] for row in rows])  # noqa: E731
    return {"ids": [row["img_id"] for row in rows], "paths": paths, "ham_labels": column("ham_label"),
            "groups": column("patient_id"), "fst_bins": column("fst_bin"),
            "fst": [row.get("fst") for row in rows],
            "manifest_sha256": rp.file_sha256(manifest_path)}


def specs_from(args: argparse.Namespace) -> dict:
    return {role: getattr(args, role) for role in pr.roles()}


def check_finals(specs: dict) -> dict:
    """Each seed's A* and B* must share code revision and exclude list (docs/28 4.1)."""
    runs = c1.run_manifests(specs)
    for seed in pr.SEEDS:
        fe.check_runs_consistent({arm: runs[pr.role(arm, seed)] for arm in pr.ARMS})
    return runs


def probe_probas(spec: str, split: dict, pad: dict, ham_index: dict, device, args) -> np.ndarray:
    """Seven-column probabilities on PAD from the C1 probe fitted on the whole C1 train."""
    _, arch, source = bank.parse_backbone_spec(f"x={spec}")
    model = bank.build_frozen_backbone(arch, source, sel.RESOLUTION, device)["model"]
    pad_labels = {i: pr.HAM_CLASSES.index(label) for i, label in zip(pad["ids"], pad["ham_labels"])}
    extract = lambda ids, index, labels: bank.extract_features(  # noqa: E731
        model, ids, index, labels, sel.RESOLUTION, device, args.batch_size, args.num_workers)
    x_fit, y_fit = extract(split["train_ids"], ham_index, split["id2label"])
    x_pad, _ = extract(pad["ids"], pad["paths"], pad_labels)
    return probes.run_probe(sel.PROBE, x_fit, y_fit, x_pad, len(pr.HAM_CLASSES))


def build_document(probas7: dict, pad: dict, provenance: dict) -> dict:
    pairs = pr.contrast_pairs()
    args = (pad["ham_labels"], pad["groups"])
    return {
        **provenance,
        "probe": sel.PROBE,
        "bootstrap": {**pr.BOOTSTRAP, "paired": True, "resampling_unit": "patient_id"},
        "readings": {name: pr.evaluate_reading(name, *args, probas7, pairs) for name in pu.READINGS},
        "fst_strata": {"reading": "primary", "exploratory": True, "min_n": pr.MIN_BIN_N,
                       "note": pu.fst_note(pu.fst_composition(pad["fst"])),
                       "bins": pr.fst_strata(*args, pad["fst_bins"], probas7, pairs)},
    }


def write_fst_table(out: Path, fst_values) -> str:
    """``<out>.fst_table.md``: images per phototype with the note computed from them."""
    composition = pu.fst_composition(list(fst_values))
    out.with_suffix(".fst_table.md").write_text(pu.fst_table_markdown(composition), encoding="utf-8")
    return pu.fst_note(composition)


def preflight(args: argparse.Namespace) -> tuple[str, str]:
    """Everything that can refuse, before any data is read."""
    if not args.single_evaluation:
        raise SystemExit("ABORT: exige --single-evaluation (PAD se lee una vez)")
    if args.out.expanduser().exists():
        raise SystemExit(f"ABORT: {args.out} ya existe; la evaluacion no se repite")
    try:
        registration = pg.check_registration(REPO)
        return registration, rp.clean_revision(REPO)
    except (pg.RegistrationError, rp.DirtyTreeError) as exc:
        raise SystemExit(f"ABORT: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registration, revision = preflight(args)
    import torch  # noqa: PLC0415

    specs = specs_from(args)
    runs = check_finals(specs)
    pad = load_pad(args.manifest.expanduser(), args.pad_root.expanduser())
    split, _ = rebuild_split(args.ham_root.expanduser())
    fe.check_val_protocol(split["train_ids"], split["val_ids"])
    if tuple(split["classes"]) != pr.HAM_CLASSES:
        raise SystemExit(f"ABORT: clases de HAM {split['classes']} != {pr.HAM_CLASSES}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ham_index = bank.index_images([args.ham_root.expanduser()])
    probas7 = {role: probe_probas(spec, split, pad, ham_index, device, args) for role, spec in specs.items()}
    provenance = {"code_revision": revision, "registration_commit": registration, "specs": specs,
                  "runs": {r: {k: run.get(k) for k in ("code_revision", "exclude_ids_sha256", "seed")}
                           for r, run in runs.items()},
                  "manifest_sha256": pad["manifest_sha256"], "dataset_doi": pf.DATASET_DOI,
                  "license": pf.LICENSE}
    document = {**build_document(probas7, pad, provenance), "environment": run_environment.snapshot(args)}
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out.with_suffix(".probas.npz"), ids=np.array(pad["ids"]), **probas7)
    print(write_fst_table(out, pad["fst"]))
    for name in pu.READINGS_WITH_VERDICT:
        print(name, document["readings"][name]["verdicts"])
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
