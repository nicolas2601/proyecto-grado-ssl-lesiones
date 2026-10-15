"""CCAM-head training and text pseudo-label export; no expert masks."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from ..common.bank import source_bank
from ..common.data import read_manifest,rgb_tensor_array,digest
from ..common.training import gate_report,seed_all,prepare_device
from ..common.checkpoints import save_checkpoint,training_lineage
from .model import ContrastiveFusion,USLSeedGenerator
from .ccam import existing_ccam_loss


def fit_ccam(manifest_path,bank_path,out,config,gate_path,*,revision,sink=None):
    gate_report(gate_path,'usl_ccam',manifest_path,config=config,stage='ccam')
    document=read_manifest(manifest_path,training=True,purpose='ccam')
    out=Path(out)
    if out.exists():raise ValueError('New CCAM output directory required')
    size=config.get('resolution',224);batch=config.get('batch_size');epochs=config.get('epochs')
    if type(batch)is not int or batch<2 or type(epochs)is not int or epochs<1:raise ValueError('Explicit CCAM budget and batch >=2 required')
    if len(document['records'])<2 or len(document['records'])%batch==1:raise ValueError('Avoid singleton final CCAM batch without dropping inputs')
    seed_all(config.get('seed',42));device=prepare_device(config)
    encoders=source_bank(bank_path,('simclr','mocov1','mocov2'))
    model=USLSeedGenerator(fusion=ContrastiveFusion(encoders)).to(device)
    class Originals(Dataset):
        def __len__(self):return len(document['records'])
        def __getitem__(self,i):return torch.from_numpy(rgb_tensor_array(document['records'][i],size))
    loader=DataLoader(Originals(),batch_size=batch,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(config.get('seed',42)))
    optimizer=torch.optim.AdamW(model.ccam.parameters(),lr=config.get('learning_rate',1e-3))
    out.mkdir(parents=True)
    for epoch in range(epochs):
        model.train();values=[]
        for images in loader:
            images=images.to(device);features=model.fusion(images)
            background=torch.sigmoid(model.ccam.background(features));foreground=1-background
            loss=existing_ccam_loss(features,foreground,config.get('ccam_alpha',.25))
            if not torch.isfinite(loss):raise ValueError('Nonfinite CCAM objective')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.ccam.parameters()):raise ValueError('Nonfinite CCAM gradient')
            optimizer.step();values.append(float(loss.detach()))
        record={'epoch':epoch+1,'mean_ccam_loss':float(np.mean(values)),'sample_count':len(document['records'])}
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if sink is not None:sink.log(record)
    return save_checkpoint(model.ccam,out/'ccam.pt',method='usl_ccam',revision=revision,manifest_sha=digest(manifest_path),
                           lineage=training_lineage(document,manifest_path,gate_path))
