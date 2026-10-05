"""E2 of docs/33: D0..D3 on PAD target_test BCC/MEL/NV with a frozen HAM-fitted probe. Single run, on the pod.

For each arm the C1 ``logreg`` probe (``c1_select_variant.PROBE``) is fitted on the whole C1 train
of HAM10000 (8,011 ids; the source labels, as docs/33 declares) and applied, frozen, to the PAD
``target_test`` images of the split written by ``pad_target_split.py``. Features use the canonical
compute: batch 64 and CUDA autocast. Primary contrast: paired D3 - D2 macro-F1, patient-grouped
bootstrap (1000 resamples, seed 20260923). ``dermssl.pad_e2`` holds the protocol: target_test is
read once through ``target_test_read.lock``, predictions are frozen with their sha256 before any
label is read.

    python scripts/pad_e2_eval.py --single-evaluation --split $P/target_split.json \
        --manifest $P/manifest.json --pad-root $P --ham-root ~/work/datos/ham10000 \
        --d1 dinov2_vitb14:$V/final_A_s42/simsiam_dinov2_backbone_final.pt \
        --d2 dinov2_vitb14:$V/D2_s42/simsiam_dinov2_backbone_final.pt \
        --d3 dinov2_vitb14:$V/D3_s42/simsiam_dinov2_backbone_final.pt \
        --out resultados/banco_completo/pad_da/pad_e2_eval.json
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

from dermssl import final_eval as fe  # noqa: E402
from dermssl import pad_e2 as e2  # noqa: E402
from dermssl import pad_target_split as ts  # noqa: E402
from dermssl import pad_ufes_files as pf  # noqa: E402
from dermssl import pad_ufes_readings as pr  # noqa: E402
from dermssl import probe_threads, probes  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402

PUBLIC_SPEC = "dinov2_vitb14:public"
PROTOCOL = ("logreg on CLS @224 fitted on the whole C1 train (source labels), frozen on PAD target_test "
            "BCC/MEL/NV; batch 64 + CUDA autocast; patient-grouped paired bootstrap")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--single-evaluation", action="store_true", help="required: target_test is read once")
    for flag in ("--split", "--manifest", "--pad-root", "--ham-root", "--out"):
        parser.add_argument(flag, type=Path, required=True)
    parser.add_argument("--d0", default=PUBLIC_SPEC, help="must be the public DINOv2")
    for arm in ("--d1", "--d2", "--d3"):
        parser.add_argument(arm, required=True, help="ARCH:SOURCE")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def preflight(args: argparse.Namespace) -> str:
    """Everything that can refuse, before any model or image is touched."""
    if not args.single_evaluation:
        raise SystemExit("ABORT: exige --single-evaluation (target_test se lee una vez)")
    if args.out.expanduser().exists():
        raise SystemExit(f"ABORT: {args.out} ya existe; la evaluacion no se repite")
    lock = ts.TargetTestLock(args.split.expanduser().parent)
    if lock.path.exists():
        raise SystemExit(f"ABORT: target_test ya se leyo ({lock.path}); la evaluacion unica no se repite")
    if args.d0.split(":", 1)[-1] != "public":
        raise SystemExit("ABORT: D0 debe ser el DINOv2 publico")
    try:
        return rp.clean_revision(REPO)
    except rp.DirtyTreeError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc


def specs_of(args: argparse.Namespace) -> dict[str, str]:
    return {arm: getattr(args, arm.lower()) for arm in e2.ARMS}


def make_predict(specs: dict, split: dict, ham_index: dict, device, args: argparse.Namespace) -> e2.Predict:
    """Per arm: bank features of HAM train and of the PAD images, one logreg fit, seven probabilities."""
    models = {arm: bank.build_frozen_backbone(*bank.parse_backbone_spec(f"x={spec}")[1:], sel.RESOLUTION,
                                               device)["model"] for arm, spec in specs.items()}

    def extract(arm, ids, index, labels):
        return bank.extract_features(models[arm], ids, index, labels, sel.RESOLUTION, device, e2.BATCH_SIZE,
                                     args.num_workers)[0]

    def predict(ids, paths):
        unlabeled = {i: 0 for i in ids}
        out = {}
        for arm in e2.ARMS:
            x_fit = extract(arm, split["train_ids"], ham_index, split["id2label"])
            y_fit = np.array([split["id2label"][i] for i in split["train_ids"]])
            x_test = extract(arm, ids, paths, unlabeled)
            out[arm] = probe_threads.call_pinned(probes.run_probe, sel.PROBE, x_fit, y_fit, x_test,
                                                 len(pr.HAM_CLASSES))
            print(f"[e2] {arm} listo")
        return out
    return predict


def provenance(specs: dict, runs: dict, split_doc: dict, revision: str, args, device, n_fit: int) -> dict:
    return {"code_revision": revision, "protocol": PROTOCOL, "probe": sel.PROBE, "specs": specs,
            "backbone_sha256": {a: rp.file_sha256(Path(s.split(":", 1)[1]).expanduser())
                                for a, s in specs.items() if not s.endswith(":public")},
            "runs": {a: {k: r.get(k) for k in ("code_revision", "exclude_ids_sha256", "seed", "max_steps")}
                     for a, r in runs.items()},
            "split_sha256": split_doc["sha256"], "split_manifest_sha256": split_doc["manifest_sha256"],
            "manifest_sha256": rp.file_sha256(args.manifest.expanduser()), "n_fit": n_fit,
            "compute": {"batch_size": e2.BATCH_SIZE, "autocast": "cuda", "device": str(device)},
            "dataset_doi": pf.DATASET_DOI, "license": pf.LICENSE}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    revision = preflight(args)
    import torch  # noqa: PLC0415

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        e2.check_canonical_compute(device.type, e2.BATCH_SIZE)
        specs = specs_of(args)
        runs = c1.run_manifests(specs)
        e2.check_d_runs(runs)
    except (e2.E2Error, OSError) as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    split, _ = rebuild_split(args.ham_root.expanduser())
    fe.check_val_protocol(split["train_ids"], split["val_ids"])
    if tuple(split["classes"]) != pr.HAM_CLASSES:
        raise SystemExit(f"ABORT: clases de HAM {split['classes']} != {pr.HAM_CLASSES}")
    ham_index = bank.index_images([args.ham_root.expanduser()])
    split_path = args.split.expanduser()
    meta = provenance(specs, runs, json.loads(split_path.read_text(encoding="utf-8")), revision, args, device,
                      len(split["train_ids"]))
    rows = json.loads(args.manifest.expanduser().read_text(encoding="utf-8"))["images"]
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    document = e2.run_single_evaluation(split_path, rows, args.pad_root.expanduser(),
                                        make_predict(specs, split, ham_index, device, args), out, provenance=meta)
    out.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    run_environment.write_sidecar(out, args, loky_inner_threads=probe_threads.PROBE_BLAS_THREADS)
    print(json.dumps(document["primary_contrast"], ensure_ascii=False))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
