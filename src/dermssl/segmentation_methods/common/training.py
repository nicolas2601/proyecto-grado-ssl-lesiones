"""Shared controlled training of own adaptations; every run needs gate evidence."""
from __future__ import annotations
import json,random,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import read_manifest,make_dataset,digest
from .checkpoints import save_checkpoint


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def prepare_device(config):
    device=torch.device(config.get('device','cpu'))
    if device.type=='cpu':return device
    if device.type!='cuda' or not torch.cuda.is_available():raise ValueError('Requested training device unavailable')
    budget=config.get('resources',{}).get('max_gpu_memory_mb')
    if not isinstance(budget,(int,float)) or budget<=0:raise ValueError('CUDA needs an explicitly measured GPU memory budget')
    free,total=torch.cuda.mem_get_info(device)
    reserve=config.get('resources',{}).get('gpu_reserve_mb',1024)
    if budget+reserve>free/(1024**2):raise ValueError('Insufficient free GPU memory for registered budget and reserve')
    torch.cuda.set_per_process_memory_fraction(budget*(1024**2)/total,device)
    return device


def gate_report(path,method,manifest_path,*,training=True):
    report=json.loads(Path(path).read_text())
    if report.get('method')!=method or report.get('input_manifest_sha256')!=digest(manifest_path):
        raise ValueError('Gate evidence does not match method and manifest')
    required=('data_verified','independent_anatomical_evaluation_resolved','resource_budget_measured','weights_provenance_verified','smoke_passed')
    if not training:required=tuple(k for k in required if k!='independent_anatomical_evaluation_resolved')
    if any(report.get(k) is not True for k in required):raise ValueError('Real training remains gated: '+','.join(k for k in required if report.get(k) is not True))
    if not report.get('verification_revision'):raise ValueError('Gate evidence revision required')
    return report


def model_for(method):
    if method=='usl_net':
        from ..usl_net.model import USLStudent
        return USLStudent()
    if method=='rpi_net':
        from ..rpi_net.model import RPINet
        return RPINet()
    if method=='tridermnet_hago':
        from ..tridermnet_hago.model import TriDermNet
        return TriDermNet()
    raise ValueError('SLED is an image-wise master; it has no gradient-training stage')


def fit(method,manifest_path,out,config,gate_path,*,sink=None,initial=None,revision=None):
    """No expert/diagnostic arguments; no split rebuilding or automatic downloads."""
    gate_report(gate_path,method,manifest_path)
    sources=4 if method=='rpi_net' else 1
    document=read_manifest(manifest_path,need_pseudos=True,sources=sources,training=True)
    out=Path(out)
    if out.exists():raise ValueError('New run directory required')
    epochs=config.get('epochs');batch=config.get('batch_size');size=config.get('resolution',224)
    if type(epochs) is not int or epochs<1 or type(batch) is not int or batch<1:raise ValueError('Explicit positive epoch/batch budget required')
    seed_all(config.get('seed',42));device=prepare_device(config)
    model=model_for(method).to(device)
    if initial is not None:model.load_state_dict(initial,strict=True)
    dataset=make_dataset(document,size,sources,paired_flip=True,seed=config.get('seed',42))
    loader=DataLoader(dataset,batch_size=batch,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(config.get('seed',42)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.get('learning_rate',1e-3),weight_decay=config.get('weight_decay',1e-5))
    hago=None
    if method=='tridermnet_hago':
        from ..tridermnet_hago.hago import HAGO
        hago=HAGO(model,optimizer,**config.get('hago',{}))
    revision=revision or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    out.mkdir(parents=True);(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    results=[];global_step=0
    for epoch in range(epochs):
        model.train();dataset.set_epoch(epoch);start=time.monotonic();losses=[]
        for item in loader:
            image,target,valid=(item[k].to(device) for k in ('image','target','valid'))
            optimizer.zero_grad(set_to_none=True)
            if method=='rpi_net':
                from ..rpi_net.loss import dynamic_uncertainty_loss
                outputs=model(image,target);loss,_=dynamic_uncertainty_loss(outputs['logits'],target,outputs['sigma'],valid,
                                                                           detach_alpha=config.get('detach_alpha',True))
            elif method=='usl_net':
                outputs=model(image);loss=model.loss(outputs,target,valid)
            else:
                from ..tridermnet_hago.loss import losses as triderm_losses
                outputs=model(image);tasks=triderm_losses(outputs,target,valid)
                loss=tasks['segmentation']+tasks['coverage']
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            if hago is not None:hago.step(tasks['segmentation'],tasks['coverage'],outputs['coverage_logits'],epoch)
            else:
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite gradient')
                optimizer.step()
            losses.append(float(loss.detach()));global_step+=1
        record={'epoch':epoch+1,'step':global_step,'mean_loss':float(np.mean(losses)),
                'elapsed_seconds':time.monotonic()-start,'sample_count':len(dataset)}
        with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        if sink is not None:sink.log(record)
        results.append(record)
    save_checkpoint(model,out/'final.pt',method=method,revision=revision,manifest_sha=digest(manifest_path),extra={'epochs':epochs,'steps':global_step})
    return results
