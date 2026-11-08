"""Build reproducible V10 lesion-classification tables from OOF readouts."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from statistics import fmean

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _fold_signature(readout: dict) -> tuple:
    signature = []
    for run in readout.get("runs", []):
        folds = run.get("folds", [])
        if not folds:
            raise ValueError("OOF fold hashes are missing")
        signature.append((run["seed"], tuple(
            (fold["fold"], fold["train_groups_sha256"], fold["test_groups_sha256"])
            for fold in folds
        )))
    if not signature:
        raise ValueError("OOF runs are missing")
    return tuple(signature)


def validate_alignment(readouts: list[dict]) -> None:
    """Reject OOF results whose cohort, class order, or grouped CV differs."""
    if not readouts:
        raise ValueError("at least one readout is required")
    reference = readouts[0]
    for readout in readouts[1:]:
        if readout.get("classes") != reference.get("classes"):
            raise ValueError("OOF class order is not aligned")
        if readout.get("image_ids") != reference.get("image_ids"):
            raise ValueError("OOF image_ids are not aligned")
        if readout.get("groups") != reference.get("groups"):
            raise ValueError("OOF groups are not aligned")
        if readout.get("settings") != reference.get("settings"):
            raise ValueError("OOF CV settings are not aligned")
        if _fold_signature(readout) != _fold_signature(reference):
            raise ValueError("OOF grouped fold hashes are not aligned")


def _metrics_from_run(readout: dict, run: dict) -> dict:
    classes = readout["classes"]
    metrics = run["metrics"]
    if metrics.get("classes") != classes:
        raise ValueError("run class order differs from readout")
    matrix = metrics["confusion_matrix"]
    if len(matrix) != len(classes) or any(len(row) != len(classes) for row in matrix):
        raise ValueError("confusion matrix shape differs from class count")
    if any(type(value) is not int or value < 0 for row in matrix for value in row):
        raise ValueError("confusion matrix counts must be nonnegative integers")
    total = sum(sum(row) for row in matrix)
    if total <= 0:
        raise ValueError("confusion matrix is empty")
    per_class = []
    for index, label in enumerate(classes):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        if support != readout["support"][label]["images"]:
            raise ValueError(f"confusion support differs for class {label}")
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class": label, "precision": precision, "recall": recall, "f1": f1})
    return {
        "classes": per_class,
        "accuracy": sum(matrix[i][i] for i in range(len(classes))) / total,
        "macro_f1": fmean(item["f1"] for item in per_class),
        "balanced_accuracy": fmean(item["recall"] for item in per_class),
        "auc_ovr_macro": fmean(metrics["auc_ovr"].values()),
        "ece": metrics["ece"],
    }


def _class_metrics(classes: list[str], results: list[dict], support: dict) -> list[dict]:
    return [{
        "class": label,
        "precision": fmean(result["classes"][index]["precision"] for result in results),
        "recall": fmean(result["classes"][index]["recall"] for result in results),
        "f1": fmean(result["classes"][index]["f1"] for result in results),
        "support_images": support[label]["images"],
        "support_groups": support[label]["groups"],
    } for index, label in enumerate(classes)]


def aggregate_method(name: str, entries: list[tuple[int | None, dict]]) -> dict:
    """Average independent encoder runs and their repeated grouped CV scores."""
    readouts = [doc for _, doc in entries]
    validate_alignment(readouts)
    classes = readouts[0]["classes"]
    seed_ids = [seed for seed, _ in entries]
    if len(seed_ids) != len(set(seed_ids)):
        raise ValueError("encoder seed is duplicated")
    seeds = [seed for seed in seed_ids if seed is not None]
    cv_seeds = [run["seed"] for run in readouts[0]["runs"]]
    results = []
    for _, readout in entries:
        run_seeds = [run["seed"] for run in readout["runs"]]
        if run_seeds != cv_seeds:
            raise ValueError("CV repetition seeds are not aligned")
        results.extend(_metrics_from_run(readout, run) for run in readout["runs"])
    summary = {key: fmean(result[key] for result in results)
               for key in ("accuracy", "macro_f1", "balanced_accuracy", "auc_ovr_macro", "ece")}
    return {
        "method": name,
        "cohort": readouts[0]["kind"],
        "classes": _class_metrics(classes, results, readouts[0]["support"]),
        **summary,
        "n_trained_seeds": len(seeds),
        "trained_seeds": seeds,
        "n_encoder_checkpoints": len(entries),
        "n_cv_repetitions_per_seed": len(cv_seeds),
        "cv_seeds": cv_seeds,
    }


def _score(value: float) -> str:
    return f"{value:.4f}"


def _leader(records: list[dict], label: str, metric: str) -> str:
    values = [(item["method"], next(row[metric] for row in item["classes"] if row["class"] == label))
              for item in records]
    best = max(value for _, value in values)
    names = sorted(name for name, value in values if value == best)
    return f"{', '.join(names)} ({_score(best)})"


def _csv_table(records: list[dict], source_hash: str, protocol_hash: str) -> str:
    columns = ["cohort", "method", "class", "precision", "recall", "f1", "support_images",
               "support_groups", "accuracy", "macro_f1", "balanced_accuracy", "auc_ovr_macro",
               "ece", "n_trained_seeds", "trained_seeds", "n_encoder_checkpoints",
               "n_cv_repetitions_per_seed", "cv_seeds", "source_sha256", "protocol_sha256"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    fields = ("cohort", "method", "accuracy", "macro_f1", "balanced_accuracy", "auc_ovr_macro", "ece",
              "n_trained_seeds", "trained_seeds", "n_encoder_checkpoints", "n_cv_repetitions_per_seed", "cv_seeds")
    for record in records:
        for row in record["classes"]:
            writer.writerow({**{key: record.get(key, "") for key in fields}, **row,
                             "trained_seeds": ";".join(map(str, record["trained_seeds"])),
                             "cv_seeds": ";".join(map(str, record["cv_seeds"])),
                             "source_sha256": source_hash, "protocol_sha256": protocol_hash})
    return output.getvalue()


def _summary_table(records: list[dict]) -> list[str]:
    lines = ["| Method | Accuracy | Macro-F1 | Balanced accuracy | Macro OVR AUC | ECE | Trained SSL seeds | Encoder checkpoints | CV repetitions per encoder |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for record in records:
        lines.append(f"| {record['method']} | {_score(record['accuracy'])} | {_score(record['macro_f1'])} | "
                     f"{_score(record['balanced_accuracy'])} | {_score(record['auc_ovr_macro'])} | "
                     f"{_score(record['ece'])} | {record['n_trained_seeds']} | "
                     f"{record['n_encoder_checkpoints']} | {record['n_cv_repetitions_per_seed']} |")
    return lines


def _class_table(records: list[dict]) -> list[str]:
    lines = ["| Method | Class | Precision | Recall | F1 | Images | Components/groups |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for record in records:
        for row in record["classes"]:
            lines.append(f"| {record['method']} | {row['class']} | {_score(row['precision'])} | "
                         f"{_score(row['recall'])} | {_score(row['f1'])} | {row['support_images']} | "
                         f"{row['support_groups']} |")
    return lines


def _leader_table(records: list[dict]) -> list[str]:
    lines = ["### Nominal class-level leaders", "", "| Class | Best nominal F1 | Best nominal recall |", "|---|---|---|"]
    for row in records[0]["classes"]:
        label = row["class"]
        lines.append(f"| {label} | {_leader(records, label, 'f1')} | {_leader(records, label, 'recall')} |")
    return lines


def _cohort_markdown(cohort: str, records: list[dict]) -> list[str]:
    lines = [f"## {cohort}", "", *_summary_table(records), "", *_class_table(records), "",
             *_leader_table(records), "", "Support counts are cohort totals; group/component counts can overlap across classes. ",
             "Trained SSL seeds counts only reported seed IDs; zero with one checkpoint means the seed ID is ",
             "not present in this evidence bundle. Leaders are exact descriptive maxima; exact ties list all method ",
             "names alphabetically. No statistical significance is implied.", ""]
    return lines


def render_tables(records: list[dict], *, source_hash: str, protocol_hash: str) -> tuple[str, str]:
    """Render aggregate CSV and Markdown without image-level identifiers."""
    if not records:
        raise ValueError("at least one aggregate record is required")
    lines = [f"source_sha256={source_hash}", f"protocol_sha256={protocol_hash}", ""]
    for cohort in sorted({record["cohort"] for record in records}):
        cohort_records = [record for record in records if record["cohort"] == cohort]
        lines.extend(_cohort_markdown(cohort, cohort_records))
    lines.extend(["Proposed MoCo result per row describes the corresponding development OOF readouts; it is "
                  "not a test-set result, ensemble, or single globally fitted deployment reader.", ""])
    return _csv_table(records, source_hash, protocol_hash), "\n".join(lines)


def _release_files(report: dict) -> dict[str, str]:
    from moco_results_inputs import make_api_report

    files = {"metrics.json": json.dumps(report, indent=2, sort_keys=True) + "\n",
             "api-report.json": json.dumps(make_api_report(report), indent=2, sort_keys=True) + "\n"}
    sections = []
    for cohort in ("PAD", "HAM"):
        records = report["cohorts"][cohort]["records"]
        csv_text, markdown_text = render_tables(records, source_hash=report["source_sha256"],
                                                 protocol_hash=report["protocol_sha256"])
        files[f"{cohort}.csv"] = csv_text
        sections.append(markdown_text.split("\n\n", 1)[1])
    files["README.md"] = (
        "# V10 lesion classification OOF results\n\n"
        "Development-only results; these tables contain aggregate metrics, not image-level predictions.\n\n"
        f"source_sha256={report['source_sha256']}\n"
        f"protocol_sha256={report['protocol_sha256']}\n\n"
        "## PAD6 and HAM7 results\n\n" + "\n\n".join(sections)
    )
    return files


def write_release_outputs(report: dict, output_dir: Path) -> dict[str, str]:
    """Write the reproducible aggregate-only JSON, CSV and Markdown artifacts."""
    files = _release_files(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (output_dir / name).write_text(content, encoding="utf-8")
    return files


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evidence-root", type=Path, default=repo / "results/moco-v10-evidence-inputs")
    parser.add_argument("--out-dir", type=Path, default=repo / "resultados/clasificacion/v10")
    parser.add_argument("--check", action="store_true", help="fail if generated outputs are stale")
    args = parser.parse_args(argv)
    from moco_results_inputs import build_release_report

    try:
        report = build_release_report(args.evidence_root)
        files = _release_files(report)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.check:
        stale = [name for name, content in files.items()
                 if not (args.out_dir / name).is_file() or (args.out_dir / name).read_text(encoding="utf-8") != content]
        if stale:
            print("stale or missing outputs: " + ", ".join(stale), file=sys.stderr)
            return 1
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (args.out_dir / name).write_text(content, encoding="utf-8")
    print(f"Generated {len(files)} aggregate-only files in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
