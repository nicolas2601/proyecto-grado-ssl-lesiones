"""CCAM-head training and text pseudo-label export; no expert masks."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from ..common.bank import source_bank
from ..common.data import read_manifest,rgb_tensor_array,digest
from ..common.training import gate_report,seed_all,prepare_device,atomic_checkpoint,rng_state,restore_rng,_json_hash
from ..common.checkpoints import save_checkpoint,training_lineage
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


def fit_ccam(manifest_path,bank_path,out,config,gate_path,*,revision,sink=None,resume=False,tracking_run_id=None):
    gate_report(gate_path,'usl_ccam',manifest_path,config=config,stage='ccam')
    document=read_manifest(manifest_path,training=True,purpose='ccam')
    out=Path(out);resume_path=out/'resume.pt'
    if resume:
        if not out.is_dir() or not resume_path.is_file():raise ValueError('--resume requires an existing CCAM directory and intact resume.pt')
    elif out.exists():raise ValueError('New CCAM output directory required')
    size=config.get('resolution',224);batch=config.get('batch_size');epochs=config.get('epochs')
    if type(batch)is not int or batch<2 or type(epochs)is not int or epochs<1:raise ValueError('Explicit CCAM budget and batch >=2 required')
    if len(document['records'])<2 or len(document['records'])%batch==1:raise ValueError('Avoid singleton final CCAM batch without dropping inputs')
    seed=config.get('seed',42);seed_all(seed);device=prepare_device(config)
    encoders=source_bank(bank_path,('simclr','mocov1','mocov2'))
    model=USLSeedGenerator(fusion=ContrastiveFusion(encoders)).to(device)
    class Originals(Dataset):
        def __len__(self):return len(document['records'])
        def __getitem__(self,i):return torch.from_numpy(rgb_tensor_array(document['records'][i],size))
    loader_generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(Originals(),batch_size=batch,shuffle=True,num_workers=0,generator=loader_generator)
    optimizer=torch.optim.AdamW(model.ccam.parameters(),lr=config.get('learning_rate',1e-3))
    identity={'method':'usl_ccam','revision':revision,'config_sha256':_json_hash(config),
              'manifest_sha256':digest(manifest_path),'gate_sha256':digest(gate_path),'tracking_run_id':tracking_run_id}
    probe_count=min(8,len(document['records']))
    with torch.no_grad():
        probe_images=torch.stack([torch.from_numpy(rgb_tensor_array(document['records'][i],size))
                                  for i in range(probe_count)]).to(device)
    if resume:
        payload=torch.load(resume_path,map_location=device,weights_only=True)
        if payload.get('identity')!=identity:raise ValueError('Resume checkpoint identity mismatch: code, config, gate, data or revision changed')
        model.ccam.load_state_dict(payload['ccam']);optimizer.load_state_dict(payload['optimizer'])
        start_epoch=int(payload['epoch']);loss_history=list(payload.get('loss_history',[]))
        stability_history=list(payload.get('stability_history',[]))
        previous_saliency=payload.get('previous_saliency')
        restore_rng(payload['rng'],loader_generator)
    else:
        out.mkdir(parents=True);start_epoch=0;loss_history=[];stability_history=[];previous_saliency=None
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
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if sink is not None:sink.log(record)
        atomic_checkpoint(resume_path,{'identity':identity,'ccam':model.ccam.state_dict(),'optimizer':optimizer.state_dict(),
            'epoch':epoch+1,'loss_history':loss_history,'stability_history':stability_history,
            'previous_saliency':previous_saliency,'rng':rng_state(loader_generator)})
    return save_checkpoint(model.ccam,out/'ccam.pt',method='usl_ccam',revision=revision,manifest_sha=digest(manifest_path),
                           lineage=training_lineage(document,manifest_path,gate_path))
