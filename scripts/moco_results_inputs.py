"""Load the explicitly approved V10 OOF evidence files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from build_moco_results import aggregate_method, validate_alignment


_V5_VARIANTS = {
    "quota4-cls4": ("MoCo quota4 CLS4 (proposed)", "quota4", "cls4", 42),
    "m0-cls": ("MoCo M0 CLS", "m0", "cls", 42),
    "m0-cls4": ("MoCo M0 CLS4", "m0", "cls4", 42),
    "m0-cls_patch": ("MoCo M0 CLS plus patch mean", "m0", "cls_patch", 42),
    "quota4-cls": ("MoCo quota4 CLS", "quota4", "cls", 42),
    "quota4-cls_patch": ("MoCo quota4 CLS plus patch mean", "quota4", "cls_patch", 42),
    "quota4-interpolate05": ("MoCo quota4 50% weight interpolation", "interpolate", "cls", 42),
    "m0-teacher": ("MoCo M0 EMA teacher", "ema-m0", "cls", 42),
    "quota4-teacher": ("MoCo quota4 EMA teacher", "ema-quota4", "cls", 42),
    "public-cls": ("Public DINOv2 (CLS)", "public", "cls", None),
}

_COMPARATORS = {
    "Legacy baseline (method identity unverified)": (
        "matched control", [("baseline-s42-pipeline-evaluation-v1", 42),
                            ("baseline-s43-evaluation-v1", 43), ("baseline-s44-evaluation-v1", 44)]),
    "VICReg": ("matched SSL control", [("vicreg-evaluation-v1", 42),
                                        ("vicreg-s43-evaluation-v1", 43), ("vicreg-s44-evaluation-v1", 44)]),
    "Barlow Twins + VICReg": ("matched SSL control", [("barlow-vicreg-evaluation-v1", 42),
                                                         ("barlow-vicreg-s43-evaluation-v1", 43),
                                                         ("barlow-vicreg-s44-evaluation-v1", 44)]),
    "SimSiam R50 (frozen bank; architecture context)": (
        "architecture context", [("bank-simsiam-r50-frozen-evaluation-v1", None)]),
}


def moco_variant_specs() -> dict[str, tuple[str, str, str, int | None]]:
    """Return the exact predeclared MoCo-v5 output variants."""
    return dict(_V5_VARIANTS)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required OOF evidence is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _variant_files(root: Path, spec: tuple, cohort: str) -> list[tuple[int | None, Path]]:
    _, job_prefix, mode, first_seed = spec
    seeds = [None] if first_seed is None else [42, 43, 44]
    files = []
    for seed in seeds:
        job = job_prefix if seed is None else f"{job_prefix}-s{seed}"
        files.append((seed, root / "jobs" / job / f"{cohort}-{mode}.json"))
    return files


def _evidence_hash(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _aggregate_cohort(cohort: str, moco_root: Path, comparator_root: Path,
                      summary: dict, source_paths: list[Path]) -> tuple[list[dict], dict]:
    records = []
    aligned_readouts = []
    for key, (label, _, _, _) in _V5_VARIANTS.items():
        entries = [(seed, _read_json(path)) for seed, path in _variant_files(moco_root, _V5_VARIANTS[key], cohort)]
        aligned_readouts.extend(doc for _, doc in entries)
        record = aggregate_method(label, entries)
        expected = summary["variants"][key][cohort]["macro_f1"]
        if abs(record["macro_f1"] - expected) > 1e-9:
            raise ValueError(f"OOF macro-F1 disagrees with source summary for {key}/{cohort}")
        record["comparison_group"] = "MoCo v5 variants" if key != "public-cls" else "public reference"
        records.append(record)
        source_paths.extend(path for _, path in _variant_files(moco_root, _V5_VARIANTS[key], cohort))
    for label, (comparison_group, variants) in _COMPARATORS.items():
        entries = []
        for folder, seed in variants:
            path = comparator_root / folder / ("PAD-readout.json" if cohort == "PAD" else "HAM-readout.json")
            doc = _read_json(path)
            entries.append((seed, doc))
            source_paths.append(path)
            aligned_readouts.append(doc)
        record = aggregate_method(label, entries)
        record["comparison_group"] = comparison_group
        records.append(record)
    validate_alignment(aligned_readouts)
    return records, {"classes": aligned_readouts[0]["classes"], "settings": aligned_readouts[0]["settings"]}


def _seed42_encoder_sha256(plan: dict) -> str:
    jobs = [job for job in plan.get("jobs", []) if job.get("id") == "quota4-s42"]
    if len(jobs) != 1:
        raise ValueError("fixed quota4 seed-42 job is missing from the source plan")
    digests = [digest for path, digest in jobs[0].get("weights", {}).items()
               if Path(path).name == "simsiam_dinov2_backbone_final.pt"]
    if len(digests) != 1 or len(digests[0]) != 64:
        raise ValueError("fixed seed-42 encoder digest is missing from the source plan")
    return digests[0]


def _reader_fit_report() -> dict:
    return {
        "source_precision": "bfloat16 CLS4, fixed seed 42 encoder",
        "selection": {"folds": 3, "cv_seeds": [20261025, 20261026, 20261027],
                      "candidate_C": [0.3, 1, 3, 10, 30, 100], "metric": "macro-F1",
                      "tie_break": "smaller C"},
        "selected_C": {"HAM7": 30, "PAD6": 30},
        "refit_scope": "all authorized development records",
        "heads": {"HAM7": {"fit_images": 1528, "fit_groups": 1152, "solver_iterations": 143},
                  "PAD6": {"fit_images": 1134, "fit_groups": 682, "solver_iterations": 144}},
    }


def make_api_report(report: dict) -> dict:
    source = {"table_receipt_sha256": report["source_sha256"],
              "plan_sha256": report["source_plan_sha256"],
              "encoder_checkpoint_sha256": report["encoder_checkpoint_sha256"]}
    table = {"markdown": "README.md", "metrics_json": "metrics.json",
             "csv": {"PAD": "PAD.csv", "HAM": "HAM.csv"}, **source}
    return {
        "release": "v10.0.0",
        "status": "aggregate development evidence; artifact and deployment status are recorded separately",
        "method": {"family": "MoCo-v3-style", "backbone": "DINOv2 ViT-B/14",
                   "training_quota": {"source": 12, "PAD_unlabeled": 4}, "feature": "CLS4",
                   "layers": [8, 9, 10, 11], "feature_dim": 3072, "seed": 42},
        "classification": report["cohorts"],
        "table": table,
        "source": source,
        "protocol": {"sha256": report["protocol_sha256"], "table_scope": report["scope"],
                     "aggregation": report["aggregation"], "reader_fit": _reader_fit_report()},
        "caveats": [
            "OOF table means average three independently trained MoCo encoders and three CV repetitions per encoder; they are not an ensemble or final-reader metrics.",
            "Final HAM7/PAD6 readers were refit on all authorized development data; no independent score is claimed for those fitted heads.",
            "PAD is transductive; HAM development was selection-exposed. Sealed HAM holdout and PAD target_test were not accessed.",
            "The legacy baseline method identity is unverified; the frozen SimSiam R50 result is architecture-context only.",
            "Model-package hashes and portable/API parity are recorded in the separate bundle manifest and release-validation evidence, not inferred from this OOF table.",
            "Research only; not a medical device and not for diagnosis or clinical decisions.",
        ],
    }


def build_release_report(evidence_root: Path) -> dict:
    """Build aggregate-only report from the approved development OOF receipts."""
    evidence_root = evidence_root.resolve()
    moco_root = evidence_root / "moco-v5-20261107"
    comparator_root = evidence_root / "comparators"
    summary_path, plan_path = moco_root / "summary.json", moco_root / "plan.json"
    summary = _read_json(summary_path)
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != summary.get("plan_sha256"):
        raise ValueError("MoCo source plan hash does not match its summary receipt")
    plan = _read_json(plan_path)
    encoder_sha256 = _seed42_encoder_sha256(plan)
    if summary.get("incomplete") or any(job.get("status") != "done" for job in summary["jobs"].values()):
        raise ValueError("MoCo source summary has incomplete jobs")
    if set(summary.get("variants", {})) != set(_V5_VARIANTS):
        raise ValueError("MoCo summary variant set differs from the registry")
    source_paths = [summary_path, plan_path]
    protocol = {}
    cohorts = {}
    for cohort in ("PAD", "HAM"):
        records, cohort_protocol = _aggregate_cohort(cohort, moco_root, comparator_root,
                                                      summary, source_paths)
        cohorts[cohort] = {"classes": cohort_protocol["classes"], "records": records}
        protocol[cohort] = cohort_protocol
    root = evidence_root.resolve()
    source_hash = _evidence_hash(source_paths, root)
    protocol_payload = {"plan_sha256": summary["plan_sha256"], "cohorts": protocol}
    protocol_hash = hashlib.sha256(json.dumps(protocol_payload, sort_keys=True,
                                               separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "release_version": "10.0.0",
        "scope": "Development out-of-fold metrics only; sealed evaluations are excluded.",
        "aggregation": "Independent encoder checkpoints and repeated grouped-CV readouts are averaged; no ensemble is formed.",
        "source_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "source_plan_sha256": summary["plan_sha256"],
        "encoder_checkpoint_sha256": encoder_sha256,
        "cohorts": cohorts,
    }
