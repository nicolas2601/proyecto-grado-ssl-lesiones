"""Explicit iterative USL/RPI adaptation; original source manifests remain immutable."""
from pathlib import Path
import json
import numpy as np
from dermssl.w3_components import USLTextPolicy,usl_text_pseudolabel
from .data import digest


def refined_manifest(document,prediction_directory,out,method,revision,*,usl_policy=None):
    if method not in {'usl_net','rpi_net'}:raise ValueError('Refinement is registered for USL and RPI only')
    out=Path(out)
    if out.exists():raise ValueError('New refinement directory required')
    prediction_directory=Path(prediction_directory)
    summary=json.loads((prediction_directory/'summary.json').read_text())
    if summary.get('method')!=method or summary.get('stage')!='predict' or not summary.get('input_manifest_sha256'):
        raise ValueError('Refinement requires a completed prediction summary for this method')
    rows=document['records'];maps=[]
    for index,row in enumerate(rows):
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
                                        'producer_revision':revision,'input_manifest_sha256':summary['input_manifest_sha256'],\n                                        'uses_expert_masks':False,'uses_diagnoses':False}
    manifest=out/'manifest.json';manifest.write_text(json.dumps(updated,indent=2)+'\n');return manifest
