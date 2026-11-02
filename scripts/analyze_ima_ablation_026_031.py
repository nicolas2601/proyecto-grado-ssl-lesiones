"""Post-hoc paired image bootstrap; no inference, training or patient-level claim."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_cielab_case_filter import STRATA, audit  # noqa: E402

PAIRS = ((42, "026", "027"), (43, "028", "029"), (44, "030", "031"))
RUNS = tuple(n for _, a, b in PAIRS for n in (a, b))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_raw(path):
    records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        name = row["image_id"]
        require(isinstance(name, str) and bool(name), "Invalid image ID")
        require(name not in records, f"Duplicate image ID: {name}")
        require(row.get("status") == "ok", f"Non-ok case: {name}")
        for field in ("dice", "sensitivity", "specificity", "reference_foreground_fraction"):
            value = float(row[field])
            require(math.isfinite(value) and 0 <= value <= 1, f"Invalid {field}: {name}")
        for field in ("image_sha256", "reference_mask_sha256"):
            value = row[field]
            require(isinstance(value, str) and len(value) == 64
                    and all(c in "0123456789abcdef" for c in value), f"Invalid {field}")
        gt, sens, spec = (row[k] for k in ("reference_foreground_fraction", "sensitivity", "specificity"))
        pred = sens * gt + (1 - spec) * (1 - gt)
        if gt + pred > 1e-12:
            require(abs(2 * sens * gt / (gt + pred) - row["dice"]) <= 1e-4,
                    f"Dice identity mismatch: {name}")
        records[name] = row
    require(bool(records), "Empty cases file")
    return records


def align_runs(runs):
    reference = next(iter(runs.values()))
    ids = sorted(reference)
    aligned = {}
    for name, rows in runs.items():
        require(set(rows) == set(reference), f"Different image IDs in {name}")
        for image_id in ids:
            for field in ("image_sha256", "reference_mask_sha256", "reference_foreground_fraction"):
                require(rows[image_id][field] == reference[image_id][field],
                        f"Mismatched {field} for {image_id} in {name}")
        aligned[name] = [rows[i] for i in ids]
    return aligned


def sign_flip_sensitivity(seed_deltas):
    values = np.asarray(seed_deltas, dtype=float)
    observed = abs(float(values.mean()))
    null = [abs(float((values * signs).mean()))
            for signs in itertools.product((-1, 1), repeat=len(values))]
    return {
        "assignments": len(null),
        "two_sided_p": sum(x >= observed - 1e-14 for x in null) / len(null),
        "assumption": "exchangeable symmetric seed-pair differences; sensitivity only",
    }


def bootstrap_contrast(differences, *, resamples, seed):
    delta = np.asarray(differences, dtype=float)
    require(delta.ndim == 2 and delta.shape[0] > 0 and delta.shape[1] == 3
            and np.isfinite(delta).all(), "Expected finite, nonempty image-by-three-seed differences")
    require(isinstance(resamples, int) and resamples > 0, "Positive resample count required")
    rng = np.random.default_rng(seed)
    sampled = np.empty((resamples, 3))
    for start in range(0, resamples, 64):
        end = min(start + 64, resamples)
        # ONE set of indices for the pair and all three seed pairs.
        indices = rng.integers(0, len(delta), size=(end - start, len(delta)))
        sampled[start:end] = delta[indices].mean(axis=1)

    def interval(values, point):
        return {"delta": float(point), "ci95": np.quantile(values, [0.025, 0.975]).tolist()}

    points = delta.mean(axis=0)
    return {
        "count": len(delta),
        "per_seed": [interval(sampled[:, i], points[i]) for i in range(3)],
        "mean_across_fixed_seeds": interval(sampled.mean(axis=1), points.mean()),
        "seed_delta_range": [float(points.min()), float(points.max())],
        "positive_seed_count": int((points > 0).sum()),
        "negative_seed_count": int((points < 0).sum()),
        "seed_sign_flip_sensitivity": sign_flip_sensitivity(points),
    }


def verify_shared_training_identity(identities):
    """Fix the common recipe, code, learning cohort and PH2 split across seeds."""
    def signature(identity):
        return {
            "source": identity["code"]["source_sha256"],
            "config": {k: v for k, v in identity["config"].items() if k not in ("seed", "device")},
            **{key: identity[key] for key in ("unlabeled_manifest_sha256", "ph2_manifest_sha256",
                                              "train_case_ids_sha256", "val_case_ids_sha256")},
        }
    signatures = [signature(identity) for identity in identities]
    require(bool(signatures) and all(x == signatures[0] for x in signatures),
            "Shared training identity differs between runs/seeds")


def verify_receipts(root):
    root = Path(root).resolve()
    receipt = read_json(root / "source_receipt.json")
    binding = read_json(root / "binding_receipt.json")
    for data in (receipt, binding):
        for rel, proof in data["files"].items():
            path = (root / rel).resolve()
            require(path.is_relative_to(root), "Receipt path escapes source directory")
            require(sha(path) == proof["sha256"], f"Source SHA mismatch: {rel}")
    training_identities = []
    for seed, a, b in PAIRS:
        profiles = []
        for number, ima_batch in ((a, 0), (b, 6)):
            run = root / number
            summary, identity = read_json(run / "summary.json"), read_json(run / "run_identity.json")
            training_identities.append(identity["identity"])
            config, gate = read_json(run / "runtime_config.json"), read_json(run / "execution_gate.json")
            claims = list((run / "claims").glob("*.json"))
            require(len(claims) == 1, f"Expected exactly one claim for {number}")
            claim = read_json(claims[0])
            require(summary["status"] == "completed" and summary["total_epochs"] == 300,
                    f"Incomplete training: {number}")
            for field in ("run_id", "identity_fingerprint"):
                require(summary[field] == identity[field] == claim[field], f"Identity mismatch: {number}")
            require(claim["gate_sha256"] == sha(run / "execution_gate.json")
                    == config["paths"]["execution_gate_sha256"], f"Gate SHA mismatch: {number}")
            require(config["profile"] == gate["profile"], f"Profile mismatch: {number}")
            profile = config["profile"]
            require(profile["seed"] == seed and profile["ima_batch"] == ima_batch,
                    f"Unexpected seed or IMA batch: {number}")
            checkpoint = receipt["checkpoint_verification"][number]
            require(checkpoint["actual_sha256"] == summary["best_teacher_checkpoint_sha256"],
                    f"Checkpoint SHA mismatch: {number}")
            require(read_json(run / "ph2_benchmark.json")["checkpoint_sha256"]
                    == summary["best_teacher_checkpoint_sha256"], f"PH2 checkpoint mismatch: {number}")
            for key in ("pool_manifest", "ph2_manifest"):
                proof = binding["runs"][number][key]
                require(proof["actual_sha256"] == proof["expected_sha256"]
                        == config["bindings"][key]["sha256"], f"Manifest SHA mismatch: {number}")
            profiles.append({k: v for k, v in profile.items()
                             if k not in ("device", "execution_claim_path", "ima_batch")})
        require(profiles[0] == profiles[1], f"Other runtime profile factors differ in seed {seed}")
    verify_shared_training_identity(training_identities)
    return {"source_receipt_sha256": sha(root / "source_receipt.json"),
            "binding_receipt_sha256": sha(root / "binding_receipt.json"),
            "shared_code_recipe_cohort_and_ph2_split_verified": True}


def analyze(root, *, resamples=5000, seed=20261102):
    root = Path(root)
    proofs = verify_receipts(root)
    aligned = align_runs({n: load_raw(root / n / "cases.jsonl") for n in RUNS})
    require(len(aligned["026"]) == 2594, "Expected the frozen 2594-case evaluation cohort")
    scores = {n: np.array([x["dice"] for x in aligned[n]]) for n in RUNS}
    areas = np.array([x["reference_foreground_fraction"] for x in aligned["026"]])
    differences = np.column_stack([scores[b] - scores[a] for _, a, b in PAIRS])
    subsets = [("overall", np.ones(len(areas), dtype=bool))]
    subsets += [(name, (areas >= low) & (areas < high)) for name, low, high in STRATA]
    contrasts = {}
    for index, (name, selected) in enumerate(subsets):
        require(bool(selected.any()), f"Empty stratum: {name}")
        contrasts[name] = bootstrap_contrast(differences[selected], resamples=resamples, seed=seed + index)
    qa = {}
    for number in RUNS:
        recomputed = audit(root / number / "cases.jsonl", 0.20)
        recorded = read_json(root / number / "cielab_filter.json")[number]
        require(recomputed == recorded, f"CIELAB audit reproduction mismatch: {number}")
        ph2 = read_json(root / number / "ph2_benchmark.json")
        qa[number] = {
            "dice": float(scores[number].mean()),
            "ph2_val_teacher_dice": ph2["group1_ph2_val_original_39"]["segmentation"]["dice_mean"],
            "degenerate_count": recorded["dropped_degenerate"],
            "retention": recorded["retention_overall"],
            "contamination_p95_before": recorded["before_filter"]["p95_skin_pool_lesion_contamination"],
            "contamination_p95_after": recorded["after_filter"]["p95_skin_pool_lesion_contamination"],
        }
    return {
        "schema_version": 1, "source_verification": proofs,
        "method": {"analysis": "post_hoc_exploratory", "bootstrap_unit": "image",
                   "resamples": resamples, "rng": "numpy.default_rng", "rng_seed": seed,
                   "analysis_source_sha256": sha(__file__),
                   "numpy_version": np.__version__, "confidence": 0.95,
                   "multiplicity_adjusted": False, "seed_order": [42, 43, 44],
                   "delta": "IMA minus control", "patient_independence_accredited": False,
                   "scope": "conditional on these six trained networks and this image cohort"},
        "contrasts": contrasts, "qa": qa,
        "pilot_decision": {
            "candidate_run": "026", "selection": "engineering baseline: control arm, fixed seed 42",
            "checkpoint": read_json(root / "source_receipt.json")["checkpoint_verification"]["026"],
            "scientific_winner_declared": False, "execution_enabled": False,
            "reason": "Preserve the control recipe while investigating the replicated small-lesion Dice loss under IMA.",
            "blockers": ["PAD skin QA without expert reference is not validated",
                         "Current classifier and ResNet34 segmenter adapters require verification",
                         "Pilot inputs and stage budget require a reviewed execution configuration"],
        },
        "limitations": [
            "Los intervalos post hoc no están ajustados por comparaciones múltiples.",
            "El bootstrap de imágenes no modela correlación por paciente/lesión ni nuevas semillas de entrenamiento.",
            "Solo hay tres semillas; la sensibilidad por cambio de signo tiene ocho asignaciones y muy baja resolución.",
            "La contaminación CIELAB usa referencia experta: es auditoría offline, no un filtro operativo para PAD.",
            "La evaluación ISIC es retrospectiva; separar cohortes de origen no acredita independencia por paciente.",
            "Cuatro logs operativos fueron truncados; los recibos de identidad no reconstruyen esos logs.",
        ],
    }


def render(report):
    lines = ["# Ablación IMA++ v8: incertidumbre y decisión para el piloto", "",
             "Reporte generado desde los casos congelados; análisis exploratorio post hoc.", "",
             "Delta = IMA++ − control. IC 95 % por bootstrap pareado de imágenes,",
             "condicional a los seis modelos. No acredita independencia por paciente.", "",
             "| Estrato | N | Δ s42 (pp) | Δ s43 (pp) | Δ s44 (pp) | Δ medio [IC 95 %] (pp) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, result in report["contrasts"].items():
        points = [x["delta"] * 100 for x in result["per_seed"]]
        combined = result["mean_across_fixed_seeds"]
        lo, hi = [x * 100 for x in combined["ci95"]]
        lines.append(f"| {name} | {result['count']} | {points[0]:+.2f} | {points[1]:+.2f} | "
                     f"{points[2]:+.2f} | {combined['delta'] * 100:+.2f} [{lo:+.2f}; {hi:+.2f}] |")
    lines += ["", "Los IC por semilla están en `analysis.json`. Se muestran todos los estratos;",
              "los intervalos no están ajustados por multiplicidad. El promedio de semillas",
              "no representa una estimación de variabilidad de futuros entrenamientos.", "",
              f"La sensibilidad por cambio de signo entre semillas tiene solo ocho asignaciones; "
              f"p bilateral global = {report['contrasts']['overall']['seed_sign_flip_sensitivity']['two_sided_p']:.2f}. "
              "Se informa como sensibilidad exploratoria, no como prueba de superioridad.", "",
              "La mejora global convive con una caída del Dice en los tres estratos menores",
              "al 5 % en las tres semillas. IMA++ reduce contaminación del pool de piel, pero",
              "la mejora de ese objetivo no implica mejor delimitación de lesiones pequeñas.", "",
              "## Controles de máscara y CIELAB", "",
              "| Run | Dice ISIC | Dice PH2 val | Degeneradas | Retención offline | Contaminación p95 antes/después |",
              "|---|---:|---:|---:|---:|---:|"]
    for n, q in report["qa"].items():
        lines.append(f"| {n} | {q['dice']:.4f} | {q['ph2_val_teacher_dice']:.4f} | "
                     f"{q['degenerate_count']} | {q['retention']:.1%} | "
                     f"{q['contamination_p95_before']:.4f} / {q['contamination_p95_after']:.4f} |")
    lines += ["", "## Decisión operativa", "",
              "Se fija 026 (control, semilla 42) como referencia para preparar el primer ejemplo",
              "técnico. No se declara un ganador científico ni una superioridad general de IMA++.",
              "La selección es una decisión de ingeniería conservadora, no una validación",
              "independiente con PH2 ni una elección optimizada sobre macro-F1 de PAD.", "",
              "Esta regla operativa se fija después de observar la ablación y antes del",
              "nuevo piloto; no se presenta como criterio científico preregistrado.", "",
              "El piloto sigue deshabilitado hasta resolver QA sin referencia experta en PAD,",
              "compatibilidad de adaptadores y una configuración de ejecución revisada.", "",
              "## Límites", ""]
    lines += ["- " + text for text in report["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    result = analyze(args.root, resamples=args.resamples, seed=args.seed)
    (args.root / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.root / "README.md").write_text(render(result), encoding="utf-8")
    print(json.dumps({"analysis": str(args.root / "analysis.json"), "pilot": result["pilot_decision"]}))


if __name__ == "__main__":
    main()
