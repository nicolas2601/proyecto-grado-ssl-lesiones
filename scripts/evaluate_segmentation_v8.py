"""Score a complete, frozen V8 export; expert masks are opened only here."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import sys
from functools import lru_cache

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dermssl.segmentation_methods.common.data import digest, read_manifest
from dermssl.segmentation_methods.common.protocol import validate_binding, object_digest, cohort_digest
from dermssl.segmentation_methods.common.checkpoints import verify_checkpoint_export

EVALUATOR_SOURCE_SHA256=digest(Path(__file__))


@lru_cache(maxsize=1)
def _metrics_module():
    directory = ROOT / 'plan-segmentacion/comun/evaluacion'
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('shared_v8_binary_metrics', directory/'evaluate_segmentation.py')
    source_sha=digest(directory/'evaluate_segmentation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if digest(directory/'evaluate_segmentation.py')!=source_sha:
        raise ValueError('Metrics source changed during import')
    module._v8_source_sha256=source_sha
    return module


def score_probability(probability, truth, *, hausdorff=False):
    probability = np.asarray(probability)
    truth = np.asarray(truth)
    if (probability.ndim != 2 or probability.size == 0 or probability.shape != truth.shape
            or not np.isfinite(probability).all() or (probability < 0).any()
            or (probability > 1).any() or not np.isin(truth, [0, 1]).all()):
        raise ValueError('Finite [0,1] probability and matching binary reference required')
    metrics = _metrics_module()
    prediction = probability >= .5
    result = metrics.compute_binary_metrics(prediction, truth.astype(bool))
    if hausdorff:
        result['hd95'] = metrics.hausdorff95(prediction, truth.astype(bool))
    return {k: float(v) if math.isfinite(v) else None for k, v in result.items()}


def exact_index(rows, expected):
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError('Explicit record list required')
    identifiers = [r.get('image_id') for r in rows]
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(expected):
        raise ValueError('IDs contain duplicates, missing records or extra records')
    return dict(zip(identifiers, rows, strict=True))


def summarize_cases(rows, *, replicates, seed):
    if not rows or type(replicates) is not int or replicates < 100 or type(seed) is not int:
        raise ValueError('Nonempty cohort and prespecified image bootstrap required')
    names = sorted(set().union(*(set(r)-{'image_id','status','reason'} for r in rows)))
    result = {'denominator': len(rows), 'failed_count': sum(r['status'] != 'ok' for r in rows),
              'bootstrap_unit': 'image', 'bootstrap_seed': seed,
              'bootstrap_replicates': replicates, 'confidence_level': .95, 'metrics': {}}
    # Resample whole image rows, including abstentions, with the same draw for all metrics.
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(replicates, len(rows)))
    for name in names:
        values = np.asarray([np.nan if r.get(name) is None else r[name] for r in rows], dtype=float)
        finite = np.isfinite(values)
        samples = values[indices]
        counts = np.isfinite(samples).sum(axis=1)
        means = np.divide(np.nansum(samples, axis=1), counts,
                          out=np.full(replicates, np.nan), where=counts > 0)
        defined = means[np.isfinite(means)]
        result['metrics'][name] = {
            'mean': float(values[finite].mean()) if finite.any() else None,
            'defined_count': int(finite.sum()), 'undefined_count': int((~finite).sum()),
            'mean_denominator': int(finite.sum()),
            'ci95': [float(v) for v in np.percentile(defined, [2.5, 97.5])] if defined.size else None,
            'undefined_bootstrap_replicates': int(replicates-defined.size)}
    return result


def evaluator_provenance(code_revision):
    if not isinstance(code_revision,str) or not re.fullmatch('[0-9a-f]{40}',code_revision):
        raise ValueError('Explicit full snapshot code revision required')
    metrics=ROOT/'plan-segmentacion/comun/evaluacion/evaluate_segmentation.py'
    if digest(Path(__file__))!=EVALUATOR_SOURCE_SHA256 or digest(metrics)!=_metrics_module()._v8_source_sha256:
        raise ValueError('Loaded evaluator/metrics source differs from current bytes')
    files={'scripts/evaluate_segmentation_v8.py':digest(Path(__file__)),
           'plan-segmentacion/comun/evaluacion/evaluate_segmentation.py':digest(metrics)}
    return {'code_revision':code_revision,'code_revision_source':'explicit_snapshot_revision',
            'evaluator_sha256':files['scripts/evaluate_segmentation_v8.py'],
            'metrics_sha256':files['plan-segmentacion/comun/evaluacion/evaluate_segmentation.py'],
            'code_sha256':object_digest(files),'code_files':files}


def evaluate_export(manifest_path, reference_path, predictions, *, hausdorff=False,code_revision=None):
    """Validate all identities before reading any expert mask or writing a score."""
    provenance=evaluator_provenance(code_revision)
    document = read_manifest(manifest_path, purpose='predict')
    if document['partition'] != 'evaluation':
        raise ValueError('Frozen evaluation partition required')
    references = json.loads(Path(reference_path).read_text())
    directory = Path(predictions)
    summary = json.loads((directory/'summary.json').read_text())
    config = json.loads((directory/'config.json').read_text())
    gate = json.loads((directory/'gate.json').read_text())
    protocol = validate_binding(document, references, summary, config, gate)
    if protocol is None:
        raise ValueError('Separate V8 evaluator requires explicit retrospective protocol')
    checkpoint_receipt=verify_checkpoint_export(config,summary,directory,summary.get('method'))
    if (summary.get('stage') != 'predict' or summary.get('export_complete') is not True
            or summary.get('input_manifest_sha256') != digest(manifest_path)
            or summary.get('config_sha256') != object_digest(config)
            or summary.get('gate_sha256') != digest(directory/'gate.json')
            or summary.get('cases_sha256') != digest(directory/'cases.jsonl')
            or summary.get('prediction_threshold') != .5 or summary.get('threshold_inclusive') is not True):
        raise ValueError('Complete hash-bound final prediction export required')
    from dermssl.segmentation_methods.common.training import gate_report
    gate_report(directory/'gate.json', summary['method'], manifest_path,
                training=False, config=config, stage='predict')
    if (references.get('schema_version') != 1 or references.get('partition') != 'evaluation'
            or references.get('evaluation_cohort_sha256') != protocol['evaluation_cohort_sha256']):
        raise ValueError('Reserved reference identity mismatch')
    expected = {r['image_id'] for r in document['records']}
    original = exact_index(document['records'], expected)
    refs = exact_index(references.get('records'), expected)
    reference_cohort=sorted((r['image_id'],r['image_sha256'],r.get('mask_sha256'))
                            for r in refs.values())
    if object_digest(reference_cohort)!=protocol['reference_cohort_sha256']:
        raise ValueError('Reference mask hashes differ from the frozen protocol')
    cases = exact_index([json.loads(line) for line in (directory/'cases.jsonl').read_text().splitlines()], expected)
    if (references.get('count') != len(expected) or summary.get('input_count') != len(expected)
            or summary.get('successful_count') != sum(r.get('status') == 'ok' for r in cases.values())
            or summary.get('abstentions') != sum(r.get('status') == 'abstained' for r in cases.values())):
        raise ValueError('Original denominator/export status counts differ')
    if cohort_digest(document['records']) != protocol['evaluation_cohort_sha256']:
        raise ValueError('Frozen cohort changed')
    size = config.get('resolution', 224)
    if type(size) is not int or size <= 0:
        raise ValueError('Positive frozen prediction resolution required')
    rows = []
    for image_id in sorted(expected):
        ref, case, image = refs[image_id], cases[image_id], original[image_id]
        if (ref.get('image_sha256') != image['image_sha256']
                or case.get('image_sha256') != image['image_sha256']
                or digest(ref['mask_path']) != ref.get('mask_sha256')):
            raise ValueError('Reference/image/export hash mismatch')
        with Image.open(ref['mask_path']) as mask:
            truth = np.asarray(mask)
            with Image.open(image['image_path']) as rgb:
                if mask.size != rgb.size:
                    raise ValueError('Original image/mask geometry mismatch')
            if truth.ndim != 2 or not np.isin(truth, [0, 1, 255]).all():
                raise ValueError('Binary single-channel lesion mask required')
            truth = np.asarray(Image.fromarray((truth > 0).astype(np.uint8)).resize(
                (size, size), Image.Resampling.NEAREST))
        if case.get('status') == 'abstained':
            row = {'image_id': image_id, 'status': 'abstained', 'reason': case.get('reason'),
                   'dice': 0., 'iou': 0., 'accuracy': None, 'sensitivity': None,
                   'specificity': None, 'pred_coverage': None, 'ref_coverage': float(truth.mean())}
            if hausdorff:
                row['hd95'] = None
        elif case.get('status') == 'ok':
            if digest(case['prediction_path']) != case.get('prediction_sha256'):
                raise ValueError('Final probability hash mismatch')
            if digest(case['valid_path']) != case.get('valid_sha256'):
                raise ValueError('Exported validity hash mismatch')
            valid = np.load(case['valid_path'], allow_pickle=False)
            if valid.shape != (size, size) or not np.all(valid == 1):
                raise ValueError('Reserved benchmark predictions must cover the whole image')
            probability = np.load(case['prediction_path'], allow_pickle=False)
            if probability.shape != (size, size):
                raise ValueError('Prediction geometry differs from frozen configuration')
            row = {'image_id': image_id, 'status': 'ok', **score_probability(probability, truth, hausdorff=hausdorff)}
        else:
            raise ValueError('Incomplete or unknown export case status')
        rows.append(row)
    report = summarize_cases(rows, replicates=protocol['bootstrap_replicates'], seed=protocol['bootstrap_seed'])
    report.update(schema_version=1, evaluation_regime=protocol['regime'],
                  evaluation_protocol=document['evaluation_protocol'],
                  patient_independence_accredited=False, original_blind_challenge_test=False,
                  scientific_efficacy_evaluation=True, method=summary['method'],
                  input_manifest_sha256=digest(manifest_path), reference_manifest_sha256=digest(reference_path),
                  export_summary_sha256=digest(directory/'summary.json'),
                  checkpoint_lineage=checkpoint_receipt,
                  prediction_threshold=.5, threshold_inclusive=True, resolution=size,
                  reference_resize='nearest', primary_failure_policy='zero_dice_iou_original_denominator',
                  per_image=rows)
    if evaluator_provenance(code_revision)!=provenance:
        raise ValueError('Evaluator or metrics source changed during evaluation')
    report.update(provenance)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wandb-project', required=True)
    parser.add_argument('--hausdorff', action='store_true')
    parser.add_argument('--code-revision',required=True,help='Full revision of the deployed ZIP snapshot')
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError('New evaluation directory required')
    if not args.wandb_project:
        raise ValueError('Explicit W&B project required')
    report = evaluate_export(args.manifest, args.references, args.predictions, hausdorff=args.hausdorff,
                             code_revision=args.code_revision)
    import wandb
    run = wandb.init(project=args.wandb_project, name=args.out.name,
                     job_type='retrospective_evaluation', config={
                         'evaluation_protocol': report['evaluation_protocol'],
                         'patient_independence_accredited': False})
    if run is None:
        raise RuntimeError('Configured W&B tracking unavailable')
    try:
        args.out.mkdir(parents=True)
        report['wandb_run_id'] = run.id
        report['wandb_url'] = run.url
        path = args.out/'evaluation.json'
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        artifact = wandb.Artifact(args.out.name, type='retrospective-evaluation')
        artifact.add_file(str(path))
        logged = run.log_artifact(artifact)
        offline=bool(getattr(run,'offline',False)) or os.environ.get('WANDB_MODE','').lower()=='offline'
        disabled=os.environ.get('WANDB_MODE','').lower()=='disabled'
        if not offline and not disabled:
            logged.wait()
        receipt = {'evaluation_sha256': digest(path),
                   'artifact': logged.qualified_name if not offline and not disabled else None,
                   'wandb_run_id': run.id, 'receipt_complete': not offline and not disabled,
                   'uploaded': not offline and not disabled,
                   'status':'offline-pending' if offline else 'disabled-pending' if disabled else 'uploaded'}
        (args.out/'wandb.receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    finally:
        run.finish()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
