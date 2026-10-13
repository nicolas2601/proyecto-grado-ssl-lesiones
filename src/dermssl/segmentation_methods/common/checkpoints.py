"""Explicit checkpoint provenance and weights-only loading; no downloads."""
from pathlib import Path
import json
import torch
from .data import digest


def load_reviewed(model,path,provenance_path,*,expected_method=None,device='cpu'):
    origin=json.loads(Path(provenance_path).read_text())
    if origin.get('uses_expert_masks') is not False or origin.get('uses_diagnoses') is not False:
        raise ValueError('Unlabelled checkpoint provenance required')
    if origin.get('checkpoint_sha256')!=digest(path) or not origin.get('producer_revision'):
        raise ValueError('Checkpoint identity incomplete or mismatched')
    if expected_method is not None and origin.get('method')!=expected_method:raise ValueError('Wrong checkpoint method')
    payload=torch.load(path,map_location=device,weights_only=True)
    model.load_state_dict(payload['model'],strict=True)
    return origin


def save_checkpoint(model,path,*,method,revision,manifest_sha,extra=None):
    path=Path(path)
    if path.exists() or path.with_suffix(path.suffix+'.json').exists():raise ValueError('Never overwrite historical checkpoint/provenance')
    torch.save({'model':model.state_dict(),'extra':extra or {}},path)
    origin={'method':method,'producer_revision':revision,'input_manifest_sha256':manifest_sha,
            'checkpoint_sha256':digest(path),'uses_expert_masks':False,'uses_diagnoses':False,'input_normalization':'unit_rgb'}
    path.with_suffix(path.suffix+'.json').write_text(json.dumps(origin,indent=2)+'\n')
    return origin
