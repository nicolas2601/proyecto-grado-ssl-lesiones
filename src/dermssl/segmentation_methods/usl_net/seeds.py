"""CCAM-head training and text pseudo-label export; no expert masks."""
from pathlib import Path
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from ..common.bank import source_bank
from ..common.data import read_manifest,rgb_tensor_array,digest
from ..common.training import gate_report,seed_all,prepare_device,atomic_checkpoint,rng_state,restore_rng,_json_hash
from ..common.checkpoints import training_lineage
from .model import ContrastiveFusion,USLSeedGenerator
from .ccam import existing_ccam_loss

# Anchored to the 30-epoch MoCo preentrenamiento precedent already used in
# this project (docs/55); not a measured CCAM-specific convergence bound.
RECOMMENDED_EPOCH_CEILING=30


def saliency_stability(previous,current):
    """Pearson correlation in [-1, 1] between two flat saliency probes; 1 means unchanged."""
    if previous.shape!=current.shape or previous.dim()!=1:raise ValueError('Matching flat saliency vectors required')
    if previous.numel()<2:raise ValueError('At least two saliency values required for correlation')
    centered_previous=previous-previous.mean();centered_current=current-current.mean()
    denominator=centered_previous.norm()*centered_current.norm()
    if denominator<=0:raise ValueError('Degenerate (constant) saliency map has no defined correlation')
    return float((centered_previous@centered_current/denominator).clamp(-1.,1.))


def select_candidate_epoch(stability_history,threshold=.98):
    """Earliest epoch (1-indexed) whose saliency stability reaches ``threshold``; None if never."""
    for index,value in enumerate(stability_history):
        if value is not None and value>=threshold:return index+1
    return None


def _atomic_json(path,value,*,jsonl=False):
    temporary=path.with_name(path.name+'.tmp')
    text=(''.join(json.dumps(row)+'\n' for row in value) if jsonl
          else json.dumps(value,indent=2)+'\n')
    temporary.write_text(text,encoding='utf-8');os.replace(temporary,path)


def _source_bank_identity(bank_path):
    document=json.loads(Path(bank_path).read_text())
    # source_bank loads these exact paths relative to cwd, not the bank directory.
    assets={origin:{kind:{'path':str(Path(source[kind]).resolve()),
                          'sha256':digest(source[kind])}
                    for kind in ('checkpoint','provenance')}
            for origin,source in document['sources'].items()}
    return {'manifest_sha256':digest(bank_path),'assets':assets}


def _export_head(path,state,*,identity,epoch,lineage,status):
    """Recoverable run-owned export; a different run must never overwrite it."""
    if path.exists():
        existing=torch.load(path,map_location='cpu',weights_only=True)
        if existing.get('extra',{}).get('identity')!=identity:
            raise ValueError('Refusing to overwrite a CCAM export from another identity')
    extra={'identity':identity,'epoch':epoch,'selection_status':status}
    payload={'model':state,'extra':extra,'lineage':lineage}
    origin={'method':'usl_ccam','protocol':'w3-four-methods-v8',
            'producer_revision':identity['revision'],'input_manifest_sha256':identity['manifest_sha256'],
            'uses_expert_masks':False,'uses_diagnoses':False,'input_normalization':'unit_rgb',
            'epoch':epoch,'selection_status':status,'source_hashes':identity['source_hashes']}
    if lineage is not None:
        payload['patient_independence_accredited']=False
        origin['lineage']=lineage;origin['patient_independence_accredited']=False
    atomic_checkpoint(path,payload);origin['checkpoint_sha256']=digest(path)
    _atomic_json(path.with_suffix(path.suffix+'.json'),origin)
    return origin


def fit_ccam(manifest_path,bank_path,out,config,gate_path,*,revision,sink=None,resume=False,tracking_run_id=None,source_hashes=None):
    gate_report(gate_path,'usl_ccam',manifest_path,config=config,stage='ccam')
    document=read_manifest(manifest_path,training=True,purpose='ccam')
    out=Path(out);resume_path=out/'resume.pt'
    if resume:
        if not out.is_dir() or not resume_path.is_file():raise ValueError('--resume requires an existing CCAM directory and intact resume.pt')
    elif out.exists():raise ValueError('New CCAM output directory required')
    size=config.get('resolution',224);batch=config.get('batch_size');epochs=config.get('epochs')
    if type(batch)is not int or batch<2 or type(epochs)is not int or epochs<1:raise ValueError('Explicit CCAM budget and batch >=2 required')
    if len(document['records'])<2 or len(document['records'])%batch==1:raise ValueError('Avoid singleton final CCAM batch without dropping inputs')
    selection=config.get('ccam_selection',{'rule':'first_stability_threshold','threshold':.98})
    threshold=selection.get('threshold')
    if (selection.get('rule')!='first_stability_threshold' or isinstance(threshold,bool)
            or not isinstance(threshold,(int,float)) or not -1<=threshold<=1):
        raise ValueError('CCAM requires first_stability_threshold with a finite Pearson threshold in [-1,1]')
    if resume and not source_hashes:raise ValueError('CCAM resume requires explicit source_hashes; legacy unbound checkpoints are not resumable here')
    seed=config.get('seed',42);seed_all(seed);device=prepare_device(config)
    bank_identity=_source_bank_identity(bank_path)
    encoders=source_bank(bank_path,('simclr','mocov1','mocov2'))
    if _source_bank_identity(bank_path)!=bank_identity:
        raise ValueError('Source bank manifest or checkpoint/provenance assets changed during loading')
    model=USLSeedGenerator(fusion=ContrastiveFusion(encoders)).to(device)
    class Originals(Dataset):
        def __len__(self):return len(document['records'])
        def __getitem__(self,i):return torch.from_numpy(rgb_tensor_array(document['records'][i],size))
    loader_generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(Originals(),batch_size=batch,shuffle=True,num_workers=0,generator=loader_generator)
    optimizer=torch.optim.AdamW(model.ccam.parameters(),lr=config.get('learning_rate',1e-3),
                               weight_decay=config.get('weight_decay',.01))
    identity={'method':'usl_ccam','revision':revision,'config_sha256':_json_hash(config),
              'manifest_sha256':digest(manifest_path),'gate_sha256':digest(gate_path),'tracking_run_id':tracking_run_id,
              'source_hashes':dict(source_hashes or {}),'source_bank_identity':bank_identity}
    probe_count=min(8,len(document['records']))
    with torch.no_grad():
        probe_images=torch.stack([torch.from_numpy(rgb_tensor_array(document['records'][i],size))
                                  for i in range(probe_count)]).to(device)
    if resume:
        # RNG states must remain CPU ByteTensors; loaders move model/optimizer state.
        payload=torch.load(resume_path,map_location='cpu',weights_only=True)
        if payload.get('identity')!=identity:raise ValueError('Resume checkpoint identity mismatch: code, config, gate, data or revision changed')
        model.ccam.load_state_dict(payload['ccam']);optimizer.load_state_dict(payload['optimizer'])
        start_epoch=int(payload['epoch']);loss_history=list(payload.get('loss_history',[]))
        stability_history=list(payload.get('stability_history',[]))
        previous_saliency=payload.get('previous_saliency')
        if previous_saliency is not None:previous_saliency=previous_saliency.to(device)
        records=list(payload['epoch_records']);selected=payload.get('selected_candidate')
        restore_rng(payload['rng'],loader_generator)
    else:
        out.mkdir(parents=True);start_epoch=0;loss_history=[];stability_history=[];previous_saliency=None
        records=[];selected=None
    # The committed checkpoint is authoritative after any logging interruption.
    _atomic_json(out/'metrics.jsonl',records,jsonl=True)
    for epoch in range(start_epoch,epochs):
        model.train();values=[]
        for images in loader:
            images=images.to(device);features=model.fusion(images)
            background=torch.sigmoid(model.ccam.background(features));foreground=1-background
            loss=existing_ccam_loss(features,foreground,config.get('ccam_alpha',.25))
            if not torch.isfinite(loss):raise ValueError('Nonfinite CCAM objective')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.ccam.parameters()):raise ValueError('Nonfinite CCAM gradient')
            optimizer.step();values.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():current_saliency=model(probe_images).detach().flatten()
        stability=None if previous_saliency is None else saliency_stability(previous_saliency,current_saliency)
        previous_saliency=current_saliency
        mean_loss=float(np.mean(values));loss_history.append(mean_loss);stability_history.append(stability)
        record={'epoch':epoch+1,'mean_ccam_loss':mean_loss,'sample_count':len(document['records']),
                'saliency_stability':stability}
        records.append(record)
        candidate_epoch=select_candidate_epoch(stability_history,threshold)
        if selected is None and candidate_epoch==epoch+1:
            selected={'epoch':epoch+1,'ccam':{name:value.detach().cpu().clone()
                       for name,value in model.ccam.state_dict().items()}}
        atomic_checkpoint(resume_path,{'identity':identity,'ccam':model.ccam.state_dict(),'optimizer':optimizer.state_dict(),
            'epoch':epoch+1,'loss_history':loss_history,'stability_history':stability_history,
            'epoch_records':records,'selected_candidate':selected,
            'previous_saliency':previous_saliency,'rng':rng_state(loader_generator)})
        _atomic_json(out/'metrics.jsonl',records,jsonl=True)
        # Logical epoch is explicit. External tracking is not an exactly-once transaction.
        if sink is not None:sink.log(record)
    lineage=training_lineage(document,manifest_path,gate_path)
    final=_export_head(out/'ccam.pt',model.ccam.state_dict(),identity=identity,epoch=epochs,
                       lineage=lineage,status='budget_final_not_selected')
    selected_origin=None
    if selected is not None:
        selected_origin=_export_head(out/'selected.ccam.pt',selected['ccam'],identity=identity,
                                     epoch=selected['epoch'],lineage=lineage,status='stability_candidate')
    _atomic_json(out/'selection.json',{'rule':selection['rule'],'threshold':threshold,
        'budget_epochs':epochs,'stopping_reason':'fixed_budget_exhausted','early_stopping':False,
        'selected_epoch':selected['epoch'] if selected is not None else None,
        'status':'stability_candidate' if selected is not None else 'threshold_not_reached',
        'final_checkpoint':{'path':'ccam.pt','sha256':final['checkpoint_sha256']},
        'selected_checkpoint':None if selected_origin is None else {
            'path':'selected.ccam.pt','sha256':selected_origin['checkpoint_sha256']},
        'scientific_convergence_verified':False,'anatomical_quality_verified':False})
    return final
