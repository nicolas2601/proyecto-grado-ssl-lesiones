"""Explicit iterative USL/RPI adaptation; original source manifests remain immutable."""
from pathlib import Path
import json
import numpy as np
from dermssl.w3_components import USLTextPolicy,usl_text_pseudolabel
from .data import digest,validate_manifest_document
from .protocol import validate_binding,cohort_digest
from .checkpoints import verify_prediction_export


def refined_manifest(document,prediction_directory,out,method,revision,*,usl_policy=None,manifest_path=None):
    if document.get('partition')!='train':raise ValueError('Refinement accepts training originals only')
    validate_manifest_document(document,need_pseudos=True,sources=4 if method=='rpi_net' else 1,
                               training=True,purpose='refine')
    protocol=validate_binding(document)
    if protocol is not None and cohort_digest(document['records']) not in protocol['learning_cohort_sha256s']:
        raise ValueError('Refinement cohort is not frozen in the protocol')
    if method not in {'usl_net','rpi_net'}:raise ValueError('Refinement is registered for USL and RPI only')
    out=Path(out)
    if out.exists():raise ValueError('New refinement directory required')
    prediction_directory=Path(prediction_directory)
    summary=json.loads((prediction_directory/'summary.json').read_text())
    validate_binding(document,summary)
    cases=None
    if protocol is not None:
        summary=verify_prediction_export(document,manifest_path,prediction_directory,method)
        if summary.get('input_cohort_sha256')!=cohort_digest(document['records']) or summary.get('export_complete') is not True:
            raise ValueError('Refinement prediction cohort differs or is incomplete')
        ledger=prediction_directory/'cases.jsonl'
        if digest(ledger)!=summary.get('cases_sha256'):raise ValueError('Refinement export ledger changed')
        cases=[json.loads(line) for line in ledger.read_text().splitlines()]
        if len(cases)!=len(document['records']):raise ValueError('Incomplete refinement prediction ledger')
    if summary.get('method')!=method or summary.get('stage')!='predict' or not summary.get('input_manifest_sha256'):
        raise ValueError('Refinement requires a completed prediction summary for this method')
    rows=document['records'];maps=[]
    for index,row in enumerate(rows):
        if cases is not None:
            case=cases[index];path=prediction_directory/f'case_{index:06d}'/'probability.npy'
            if (case.get('image_id')!=row['image_id'] or case.get('image_sha256')!=row['image_sha256']
                    or case.get('status')!='ok' or Path(case.get('prediction_path','')).resolve()!=path.resolve()
                    or digest(path)!=case.get('prediction_sha256')):
                raise ValueError('Refinement probability identity/hash mismatch')
        probability=np.load(prediction_directory/f'case_{index:06d}'/'probability.npy',allow_pickle=False)
        if probability.ndim!=2 or not np.isfinite(probability).all() or ((probability<0)|(probability>1)).any():raise ValueError('Valid aligned prediction probability required')
        if method=='usl_net':value,_=usl_text_pseudolabel(probability,USLTextPolicy(**(usl_policy or {})))
        else:value=probability.astype(np.float32) # registered replacement of P2 ONLY, retain independent P1/P3/P4
        maps.append(value)
    out.mkdir(parents=True);updated=json.loads(json.dumps(document))
    for index,(row,value) in enumerate(zip(updated['records'],maps,strict=True)):
        if 'valid_path' in row:
            valid=np.load(row['valid_path'],allow_pickle=False)
            if valid.shape!=value.shape:raise ValueError('Refinement valid geometry differs; use fixed registered resolution')
            value=np.where(valid,value,.5)
        path=out/f'pseudo_{index:06d}.npy';np.save(path,value)
        source=0 if method=='usl_net' else 1
        row['pseudo_paths'][source]=str(path.resolve());row['pseudo_sha256s'][source]=digest(path)
        row['pseudo_provenance'][source]={'method':method,'stage':'iterative_refinement',
                                        'producer_revision':revision,'input_manifest_sha256':summary['input_manifest_sha256'],
                                        'uses_expert_masks':False,'uses_diagnoses':False}
    manifest=out/'manifest.json';manifest.write_text(json.dumps(updated,indent=2)+'\n');return manifest
