"""Shared controlled training of own adaptations; every run needs gate evidence."""
from __future__ import annotations
import hashlib,json,os,random,signal,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Sampler
from .data import read_manifest,make_dataset,digest
from .checkpoints import save_checkpoint

_STOP_REQUESTED=False


def request_stop(signum=None,frame=None):
    global _STOP_REQUESTED
    _STOP_REQUESTED=True


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


def _json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()


def _state_hash(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _rng_state(loader_generator):
    ns=np.random.get_state()
    return {'python':random.getstate(),'numpy':{'name':ns[0],'keys':torch.from_numpy(ns[1].copy()),
            'position':int(ns[2]),'has_gauss':int(ns[3]),'cached_gaussian':float(ns[4])},
            'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'loader':loader_generator.get_state()}


def _restore_rng(state,loader_generator):
    random.setstate(tuple(state['python']))
    ns=state['numpy']
    np.random.set_state((ns['name'],ns['keys'].cpu().numpy().astype(np.uint32),int(ns['position']),
                         int(ns['has_gauss']),float(ns['cached_gaussian'])))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if torch.cuda.is_available() and state.get('torch_cuda'):torch.cuda.set_rng_state_all(state['torch_cuda'])
    loader_generator.set_state(state['loader'].cpu())


def _atomic_checkpoint(path,payload):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    torch.save(payload,tmp)
    os.replace(tmp,path)


class ResumableRandomSampler(Sampler):
    """Epoch permutation with a persisted cursor; paired augmentation is index/epoch seeded."""
    def __init__(self,size,seed):
        self.size=size;self.seed=seed;self.epoch=None;self.permutation=None;self.cursor=0
    def set_epoch(self,epoch):
        if self.epoch!=epoch:
            generator=torch.Generator().manual_seed(self.seed+epoch)
            self.permutation=torch.randperm(self.size,generator=generator)
            self.cursor=0;self.epoch=epoch
    def __iter__(self):
        while self.cursor<self.size:
            value=int(self.permutation[self.cursor]);self.cursor+=1
            yield value
    def __len__(self):return self.size-self.cursor
    def state_dict(self):return {'epoch':self.epoch,'permutation':self.permutation,'cursor':self.cursor}
    def load_state_dict(self,state):
        permutation=state.get('permutation')
        if not torch.is_tensor(permutation) or permutation.numel()!=self.size or sorted(permutation.tolist())!=list(range(self.size)):
            raise ValueError('Checkpoint sampler permutation is invalid')
        cursor=state.get('cursor')
        if type(cursor)is not int or not 0<=cursor<=self.size:raise ValueError('Checkpoint sampler cursor is invalid')
        self.epoch=state.get('epoch');self.permutation=permutation.long();self.cursor=cursor


def fit(method,manifest_path,out,config,gate_path,*,sink=None,initial=None,revision=None,resume=False,tracking_run_id=None):
    """No expert/diagnostic arguments; no split rebuilding or automatic downloads."""
    global _STOP_REQUESTED
    _STOP_REQUESTED=False
    gate_report(gate_path,method,manifest_path)
    sources=4 if method=='rpi_net' else 1
    document=read_manifest(manifest_path,need_pseudos=True,sources=sources,training=True)
    out=Path(out);resume_path=out/'resume.pt'
    if resume:
        if not out.is_dir() or not resume_path.is_file():raise ValueError('--resume requires an existing fit directory and intact resume.pt')
    elif out.exists():raise ValueError('New run directory required; use --resume only with an intact checkpoint')
    epochs=config.get('epochs');batch=config.get('batch_size');size=config.get('resolution',224)
    if type(epochs)is not int or epochs<1 or type(batch)is not int or batch<1:raise ValueError('Explicit positive epoch/batch budget required')
    seed=config.get('seed',42);seed_all(seed);device=prepare_device(config)
    model=model_for(method).to(device)
    if initial is not None:model.load_state_dict(initial,strict=True)
    initial_hash=_state_hash(model.state_dict())
    dataset=make_dataset(document,size,sources,paired_flip=True,seed=seed)
    sampler=ResumableRandomSampler(len(dataset),seed)
    loader_generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(dataset,batch_size=batch,sampler=sampler,num_workers=0,generator=loader_generator)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.get('learning_rate',1e-3),weight_decay=config.get('weight_decay',1e-5))
    hago=None
    if method=='tridermnet_hago':
        from ..tridermnet_hago.hago import HAGO
        hago=HAGO(model,optimizer,**config.get('hago',{}))
    revision=revision or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    identity={'method':method,'revision':revision,'config_sha256':_json_hash(config),
              'manifest_sha256':digest(manifest_path),'gate_sha256':digest(gate_path),
              'initialization_sha256':initial_hash,'architecture':model.__class__.__module__+'.'+model.__class__.__qualname__,
              'tracking_run_id':tracking_run_id}
    checkpoint_every=config.get('checkpoint_every_steps',100)
    if type(checkpoint_every)is not int or checkpoint_every<1:raise ValueError('checkpoint_every_steps must be a positive integer')
    if resume:
        payload=torch.load(resume_path,map_location=device,weights_only=True)
        if payload.get('identity')!=identity:raise ValueError('Resume checkpoint identity mismatch: code, config, gate, data, initialization or architecture changed')
        model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer'])
        if hago is not None:
            if payload.get('hago_step_number') is None:raise ValueError('HAGO state missing from resume checkpoint')
            hago.step_number=int(payload['hago_step_number'])
        epoch=int(payload['epoch']);global_step=int(payload['global_step'])
        sampler.load_state_dict(payload['sampler']);_restore_rng(payload['rng'],loader_generator)
        losses=list(payload.get('epoch_losses',[]));results=list(payload.get('results',[]))
        if epoch<0 or epoch>epochs:raise ValueError('Checkpoint epoch exceeds configured budget')
    else:
        out.mkdir(parents=True);(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        epoch=0;global_step=0;losses=[];results=[]
    def save_resume():
        _atomic_checkpoint(resume_path,{'identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'epoch':epoch,'global_step':global_step,'sampler':sampler.state_dict(),'epoch_losses':losses,
            'results':results,'rng':_rng_state(loader_generator),
            'hago_step_number':None if hago is None else hago.step_number})
    if not resume:
        sampler.set_epoch(epoch)
        save_resume()
    while epoch<epochs:
        model.train();dataset.set_epoch(epoch);start=time.monotonic()
        sampler.set_epoch(epoch)
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
            if global_step%checkpoint_every==0:save_resume()
            if _STOP_REQUESTED:
                save_resume()
                raise InterruptedError('Training stopped at a safe batch boundary; resume.pt is current')
        record={'epoch':epoch+1,'step':global_step,'mean_loss':float(np.mean(losses)),
                'elapsed_seconds':time.monotonic()-start,'sample_count':len(dataset)}
        results.append(record);epoch+=1;losses=[]
        save_resume()
        metrics_tmp=out/'metrics.jsonl.tmp'
        metrics_tmp.write_text(''.join(json.dumps(row)+'\n' for row in results))
        os.replace(metrics_tmp,out/'metrics.jsonl')
        if sink is not None:sink.log(record)
    save_checkpoint(model,out/'final.pt',method=method,revision=revision,manifest_sha=digest(manifest_path),extra={'epochs':epochs,'steps':global_step})
    return results
