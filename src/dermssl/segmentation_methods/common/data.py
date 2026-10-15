"""Frozen eligible manifests, hashed RGB/pseudos, and paired geometry for v8."""
from __future__ import annotations
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
from .protocol import validate_binding, cohort_digest


FORBIDDEN={'diagnosis','diagnostic_label','class_label','expert_mask','ground_truth','phototype','ita'}
ALLOWED={'image_id','dataset','image_path','image_sha256','pseudo_paths','pseudo_sha256s',
         'pseudo_provenance','valid_path','valid_sha256'}


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read_manifest(path, *, need_pseudos=False, sources=1, training=False, expected_pseudo_method=None, purpose=None):
    document=json.loads(Path(path).read_text())
    return validate_manifest_document(document,need_pseudos=need_pseudos,sources=sources,
                                      training=training,expected_pseudo_method=expected_pseudo_method,purpose=purpose)


def validate_manifest_document(document, *, need_pseudos=False, sources=1, training=False, expected_pseudo_method=None, purpose=None):
    if not isinstance(document,dict) or set(document)-{'schema_version','partition','count','eligibility','records','evaluation_protocol'} or document.get('schema_version')!=1:
        raise ValueError('Versioned eligible-pool manifest required; no directory scan or implicit resplit')
    protocol=validate_binding(document)
    reserved=document.get('partition')=='evaluation'
    if reserved and (protocol is None or purpose!='predict' or training or need_pseudos):
        raise ValueError('Reserved evaluation images are allowed only for final prediction')
    if (not reserved and document.get('partition') not in {'train','development'}) or (training and document['partition']!='train'):
        raise ValueError('Only frozen training/development inputs allowed; target_test/evaluation excluded')
    proof=document.get('eligibility',{})
    groups_ok=proof.get('patient_lesion_groups_preserved') is True
    if protocol is not None:
        groups_ok=(proof.get('patient_independence_accredited') is False
                   and proof.get('patient_lesion_groups_preserved') is False)
    if proof.get('verified') is not True or proof.get('excludes_target_test') is not True or not groups_ok or proof.get('deduplication_preserved') is not True:
        raise ValueError('Explicit verified eligibility, exclusions and preserved groups required')
    for key in ('source_manifest_sha256','exclusions_sha256','verification_revision'):
        if not isinstance(proof.get(key),str) or not proof[key]: raise ValueError('Eligibility provenance incomplete')
    rows=document.get('records',[])
    if not rows or document.get('count')!=len(rows) or len({r.get('image_id') for r in rows})!=len(rows):
        raise ValueError('Nonempty fixed count and unique original IDs required')
    for r in rows:
        if reserved and set(r)-{'image_id','dataset','image_path','image_sha256'}:
            raise ValueError('Reserved prediction manifest permits original images only; no expert masks or pseudos')
        if set(r)-ALLOWED or not {'image_id','dataset','image_path','image_sha256'}<=set(r):
            raise ValueError('Only original-image and pseudo provenance fields allowed')
        if not isinstance(r['image_id'],str) or not r['image_id']:raise ValueError('Nonempty string image ID required')
        if digest(r['image_path'])!=r['image_sha256']:raise ValueError('Original-image hash mismatch')
        if 'valid_path' in r and digest(r['valid_path'])!=r.get('valid_sha256'):raise ValueError('Validity hash mismatch')
        if need_pseudos:
            paths=r.get('pseudo_paths',[]); hashes=r.get('pseudo_sha256s',[])
            if len(paths)!=sources or len(hashes)!=sources:raise ValueError('Required pseudo sources missing')
            if any(digest(p)!=h for p,h in zip(paths,hashes,strict=True)):raise ValueError('Pseudo-label hash mismatch')
            origins=r.get('pseudo_provenance',[])
            if len(origins)!=sources:raise ValueError('Every pseudo source needs provenance')
            for origin in origins:
                if expected_pseudo_method is not None and origin.get('method')!=expected_pseudo_method:
                    raise ValueError('Pseudo-label producer differs from the frozen method configuration')
                if origin.get('uses_expert_masks') is not False or origin.get('uses_diagnoses') is not False:
                    raise ValueError('Expert/diagnostic supervision prohibited in W3')
                if not origin.get('producer_revision') or not origin.get('method'):raise ValueError('Pseudo producer identity missing')
    if reserved and cohort_digest(rows)!=protocol['evaluation_cohort_sha256']:
        raise ValueError('Evaluation cohort differs from the frozen protocol')
    if protocol is not None and not reserved:
        if cohort_digest(rows) not in protocol['learning_cohort_sha256s']:
            raise ValueError('Learning cohort is not frozen in the reviewed protocol')
        reserved_ids={r['image_id'] for r in protocol['reserved_images']}
        reserved_hashes={r['image_sha256'] for r in protocol['reserved_images']}
        if any(r['image_id'] in reserved_ids or r['image_sha256'] in reserved_hashes for r in rows):
            raise ValueError('Reserved image ID or identical content appears in learning')
    return document


def rgb_tensor_array(row,size):
    if digest(row['image_path'])!=row['image_sha256']:raise ValueError('Original changed during run')
    with Image.open(row['image_path']) as image:
        rgb=np.asarray(image.convert('RGB').resize((size,size),resample=Image.Resampling.BILINEAR),dtype=np.float32)/255
    return rgb.transpose(2,0,1).copy()


def _array(path):
    value=np.load(path,allow_pickle=False)
    if not isinstance(value,np.ndarray) or value.ndim!=2 or not np.isfinite(value).all():
        raise ValueError('Finite numeric 2D NPY maps required')
    return value


def pseudo_array(row,size,sources):
    maps=[]
    for path,sha in zip(row['pseudo_paths'],row['pseudo_sha256s'],strict=True):
        if digest(path)!=sha:raise ValueError('Pseudo changed during run')
        value=_array(path).astype(np.float32)
        if sources==1 and not np.isin(value,[0,.5,1]).all():raise ValueError('Single-source pseudo must encode 0/.5/1')
        if (value<0).any() or (value>1).any():raise ValueError('Pseudo targets must be in [0,1]')
        maps.append(np.asarray(Image.fromarray(value).resize((size,size),Image.Resampling.NEAREST)))
    if len(maps)!=sources:raise ValueError('Incorrect number of pseudo maps')
    return np.stack(maps).copy()


def valid_array(row,size):
    if 'valid_path' not in row:return np.ones((1,size,size),dtype=np.float32)
    if digest(row['valid_path'])!=row['valid_sha256']:raise ValueError('Validity changed during run')
    value=_array(row['valid_path'])
    if not np.isin(value,[0,1]).all():raise ValueError('Binary validity map required')
    value=np.asarray(Image.fromarray(value.astype(np.uint8)).resize((size,size),Image.Resampling.NEAREST))
    return value[None].astype(np.float32)


def make_dataset(document,size=224,sources=1,paired_flip=False,seed=42):
    import torch
    from torch.utils.data import Dataset
    class FrozenPseudoDataset(Dataset):
        def __init__(self):self.epoch=0
        def __len__(self):return len(document['records'])
        def set_epoch(self,epoch):self.epoch=epoch
        def __getitem__(self,index):
            r=document['records'][index];rgb=rgb_tensor_array(r,size)
            target=pseudo_array(r,size,sources);valid=valid_array(r,size)
            # RNG by epoch/index: same paired geometry, no global RNG mutation.
            rng=np.random.default_rng(np.random.SeedSequence([seed,self.epoch,index]))
            if paired_flip:
                for axis in (-1,-2):
                    if rng.random()<.5:
                        rgb=np.flip(rgb,axis).copy();target=np.flip(target,axis).copy();valid=np.flip(valid,axis).copy()
            return {'image':torch.from_numpy(rgb),'target':torch.from_numpy(target),'valid':torch.from_numpy(valid)}
    return FrozenPseudoDataset()
