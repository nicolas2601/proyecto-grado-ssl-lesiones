"""Explicit checkpoint provenance and weights-only loading; no downloads."""
from pathlib import Path
import json
import re
import torch
from .data import digest
from .protocol import validate_binding,cohort_digest,object_digest


def verify_checkpoint_export(config,summary,directory,method):
    if method=='sled':
        if summary.get('checkpoint_kind')!='image_wise_no_checkpoint' or 'checkpoint_lineage_sha256' in summary:
            raise ValueError('SLED export must explicitly declare image-wise execution without a checkpoint')
        return None
    if summary.get('checkpoint_kind')!='trained':
        raise ValueError('Trained-method export requires checkpoint lineage receipt')
    requirements=checkpoint_requirements(config)
    if not requirements:raise ValueError('Retrospective checkpoint export requires protocol lineage')
    receipt_path=Path(directory)/'checkpoint.lineage.json'
    if not receipt_path.is_file() or digest(receipt_path)!=summary.get('checkpoint_lineage_sha256'):
        raise ValueError('Exported checkpoint lineage receipt missing or changed')
    receipt=json.loads(receipt_path.read_text())
    if (receipt.get('method')!=method or receipt.get('lineage')!=requirements['expected_lineage']
            or receipt.get('checkpoint_sha256')!=requirements['expected_checkpoint_sha256']
            or receipt.get('provenance_sha256')!=requirements['expected_provenance_sha256']
            or receipt.get('patient_independence_accredited') is not False):
        raise ValueError('Export lineage receipt differs from reviewed checkpoint configuration')
    origin,_=checkpoint_identity(config['checkpoint'],config['checkpoint_provenance'],**requirements)
    if origin.get('method')!=method:raise ValueError('Export checkpoint method mismatch')
    return receipt


def verify_prediction_export(document,manifest_path,directory,method):
    """Bind a prediction used for learning to its reviewed input and model receipt."""
    from .data import read_manifest
    from .training import gate_report
    if manifest_path is None:raise ValueError('Original manifest path required for retrospective refinement')
    if read_manifest(manifest_path,purpose='predict')!=document:
        raise ValueError('Refinement document differs from its original frozen manifest')
    directory=Path(directory)
    summary=json.loads((directory/'summary.json').read_text())
    config=json.loads((directory/'config.json').read_text())
    gate=json.loads((directory/'gate.json').read_text())
    validate_binding(document,summary,config,gate)
    if (summary.get('method')!=method or summary.get('stage')!='predict'
            or summary.get('export_complete') is not True
            or summary.get('input_manifest_sha256')!=digest(manifest_path)
            or summary.get('input_cohort_sha256')!=cohort_digest(document['records'])
            or summary.get('input_count')!=len(document['records'])
            or summary.get('successful_count')!=len(document['records']) or summary.get('abstentions')!=0
            or summary.get('config_sha256')!=object_digest(config)
            or summary.get('gate_sha256')!=digest(directory/'gate.json')
            or summary.get('cases_sha256')!=digest(directory/'cases.jsonl')):
        raise ValueError('Prediction export identity or hashes mismatch')
    gate_report(directory/'gate.json',method,manifest_path,training=False,config=config,stage='predict')
    verify_checkpoint_export(config,summary,directory,method)
    return summary


def training_lineage(document,manifest_path,gate_path):
    if validate_binding(document) is None:return None
    return {'protocol_sha256':document['evaluation_protocol']['sha256'],
            'training_manifest_sha256':digest(manifest_path),'training_gate_sha256':digest(gate_path),
            'training_cohort_sha256':cohort_digest(document['records'])}


def checkpoint_requirements(config,prefix='checkpoint'):
    protocol=validate_binding(config)
    if protocol is None:return {}
    lineage=config.get(prefix+'_lineage')
    validate_lineage(lineage)
    if (lineage['protocol_sha256']!=config['evaluation_protocol']['sha256']
            or lineage['training_cohort_sha256'] not in protocol['learning_cohort_sha256s']):
        raise ValueError('Checkpoint lineage differs from the reviewed protocol/learning cohort')
    checkpoint_sha=config.get(prefix+'_sha256');origin_sha=config.get(prefix+'_provenance_sha256')
    for value in (checkpoint_sha,origin_sha):
        if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
            raise ValueError('Reviewed checkpoint and provenance SHA-256 required')
    return {'expected_lineage':lineage,'expected_checkpoint_sha256':checkpoint_sha,
            'expected_provenance_sha256':origin_sha}


def validate_lineage(lineage):
    required={'protocol_sha256','training_manifest_sha256','training_gate_sha256','training_cohort_sha256'}
    if (not isinstance(lineage,dict) or set(lineage)!=required
            or any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in lineage.values())):
        raise ValueError('Complete checkpoint protocol/training manifest/gate/cohort lineage required')


def checkpoint_identity(path,provenance_path,*,expected_lineage=None,expected_checkpoint_sha256=None,
                        expected_provenance_sha256=None,device='cpu'):
    if expected_checkpoint_sha256 is not None and digest(path)!=expected_checkpoint_sha256:
        raise ValueError('Checkpoint differs from reviewed configuration')
    if expected_provenance_sha256 is not None and digest(provenance_path)!=expected_provenance_sha256:
        raise ValueError('Checkpoint provenance differs from reviewed configuration')
    origin=json.loads(Path(provenance_path).read_text())
    if origin.get('uses_expert_masks') is not False or origin.get('uses_diagnoses') is not False:
        raise ValueError('Unlabelled checkpoint provenance required')
    if origin.get('checkpoint_sha256')!=digest(path) or not origin.get('producer_revision'):
        raise ValueError('Checkpoint identity incomplete or mismatched')
    payload=torch.load(path,map_location=device,weights_only=True)
    if expected_lineage is not None:
        validate_lineage(expected_lineage)
        if (origin.get('lineage')!=expected_lineage or payload.get('lineage')!=expected_lineage
                or origin.get('input_manifest_sha256')!=expected_lineage['training_manifest_sha256']
                or origin.get('patient_independence_accredited') is not False
                or payload.get('patient_independence_accredited') is not False):
            raise ValueError('Checkpoint/sidecar embedded training lineage mismatch')
    elif origin.get('lineage') is not None or payload.get('lineage') is not None:
        if origin.get('lineage')!=payload.get('lineage'):
            raise ValueError('Checkpoint/sidecar lineage differs')
    return origin,payload


def load_reviewed(model,path,provenance_path,*,expected_method=None,required_protocol=None,device='cpu',
                  expected_lineage=None,expected_checkpoint_sha256=None,expected_provenance_sha256=None):
    origin,payload=checkpoint_identity(path,provenance_path,device=device,
                    expected_lineage=expected_lineage,expected_checkpoint_sha256=expected_checkpoint_sha256,
                    expected_provenance_sha256=expected_provenance_sha256)
    if expected_method is not None and origin.get('method')!=expected_method:raise ValueError('Wrong checkpoint method')
    if required_protocol is not None and origin.get('protocol')!=required_protocol:
        raise ValueError('Checkpoint does not carry the required training protocol lineage')
    model.load_state_dict(payload['model'],strict=True)
    return origin


def save_checkpoint(model,path,*,method,revision,manifest_sha,extra=None,lineage=None):
    path=Path(path)
    if path.exists() or path.with_suffix(path.suffix+'.json').exists():raise ValueError('Never overwrite historical checkpoint/provenance')
    if lineage is not None:
        validate_lineage(lineage)
        if lineage['training_manifest_sha256']!=manifest_sha:raise ValueError('Checkpoint manifest lineage mismatch')
    payload={'model':model.state_dict(),'extra':extra or {},'lineage':lineage}
    if lineage is not None:payload['patient_independence_accredited']=False
    torch.save(payload,path)
    origin={'method':method,'protocol':'w3-four-methods-v8','producer_revision':revision,'input_manifest_sha256':manifest_sha,
            'checkpoint_sha256':digest(path),'uses_expert_masks':False,'uses_diagnoses':False,'input_normalization':'unit_rgb'}
    if lineage is not None:
        origin['lineage']=lineage;origin['patient_independence_accredited']=False
    path.with_suffix(path.suffix+'.json').write_text(json.dumps(origin,indent=2)+'\n')
    return origin
