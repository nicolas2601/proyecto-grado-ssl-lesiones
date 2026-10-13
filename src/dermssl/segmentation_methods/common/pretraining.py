"""Local label-free source-bank/CCAM pretraining, gated independently of imports."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from .data import read_manifest,rgb_tensor_array,digest
from .training import gate_report,seed_all,prepare_device
from .checkpoints import save_checkpoint
from .ssl import ContrastiveEncoder,MomentumContrast,MoCoV3,simclr_loss
from .dino import DinoDistillation


def build_pretrainer(variant):
    if variant=='dino':return DinoDistillation()
    if variant=='simclr':return ContrastiveEncoder('simclr')
    encoder=ContrastiveEncoder('mocov1' if variant=='mocov1' else 'mocov2')
    if variant in {'mocov1','mocov2'}:return MomentumContrast(encoder)
    if variant=='mocov3':return MoCoV3(encoder)
    raise ValueError('Supported SSL origins: simclr/mocov1/mocov2/mocov3/dino')


def local_pretrain(variant,manifest_path,out,config,gate_path,*,revision,sink=None):
    gate_report(gate_path,'ssl_'+variant,manifest_path)
    document=read_manifest(manifest_path,training=True)
    size=config.get('resolution',224);seed=config.get('seed',42);epochs=config.get('epochs');batch=config.get('batch_size')
    if type(epochs)is not int or epochs<1 or type(batch)is not int or batch<2:raise ValueError('Explicit positive SSL budget and batch >=2 required')
    if len(document['records'])<2 or len(document['records'])%batch==1:
        raise ValueError('Adjust batch budget to avoid a singleton contrastive batch without dropping inputs')
    out=Path(out)
    if out.exists():raise ValueError('New SSL output directory required')
    seed_all(seed);device=prepare_device(config);model=build_pretrainer(variant).to(device)
    class Views(Dataset):
        epoch=0
        def __len__(self):return len(document['records'])
        def __getitem__(self,index):
            rgb=rgb_tensor_array(document['records'][index],size)
            rng=np.random.default_rng(np.random.SeedSequence([seed,self.epoch,index]))
            views=[]
            for _ in range(2):
                # Local augmentation adaptation: flip + color/brightness, geometry fixed.
                # No inferred recreation of original multi-crop/hyperparameters.
                value=rgb.copy()
                if rng.random()<.5:value=value[...,::-1].copy()
                gain=rng.uniform(.8,1.2,(3,1,1));shift=rng.uniform(-.1,.1)
                value=np.clip(value*gain+shift,0,1).astype(np.float32)
                views.append(torch.from_numpy(value))
            return tuple(views)
    dataset=Views();loader=DataLoader(dataset,batch_size=batch,shuffle=True,num_workers=0,
                                    generator=torch.Generator().manual_seed(seed))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config.get('learning_rate',1e-3))
    out.mkdir(parents=True)
    for epoch in range(epochs):
        dataset.epoch=epoch;model.train();values=[]
        for first,second in loader:
            first=first.to(device);second=second.to(device);optimizer.zero_grad(set_to_none=True)
            loss=simclr_loss(model(first),model(second)) if variant=='simclr' else model(first,second)
            if not torch.isfinite(loss):raise ValueError('Nonfinite SSL loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite SSL gradient')
            optimizer.step();values.append(float(loss.detach()))
        record={'epoch':epoch+1,'mean_ssl_loss':float(np.mean(values)),'sample_count':len(dataset)}
        if sink is not None:sink.log(record)
        import json
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    # Export the feature encoder, not a projector or momentum queue state.
    backbone=model.backbone if variant in {'simclr','dino'} else model.online.backbone
    return save_checkpoint(backbone,out/'encoder.pt',method='ssl_'+variant,revision=revision,manifest_sha=digest(manifest_path),extra={'epochs':epochs})
