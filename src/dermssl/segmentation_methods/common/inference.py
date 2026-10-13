"""Manifest-wide seed/prediction export; failures retained, no replacement images."""
from __future__ import annotations
from pathlib import Path
import json,time
import numpy as np
import torch
from .data import read_manifest,rgb_tensor_array,valid_array,digest
from .training import gate_report,model_for,prepare_device
from .bank import source_bank
from .checkpoints import load_reviewed
from dermssl.w3_components import DegenerateCase,USLTextPolicy,usl_text_pseudolabel


def _usl_generator(config):
    from ..usl_net.model import USLSeedGenerator,ContrastiveFusion
    model=USLSeedGenerator(ContrastiveFusion(source_bank(config['source_bank'],('simclr','mocov1','mocov2'))))
    load_reviewed(model.ccam,config['ccam_checkpoint'],config['ccam_provenance'],expected_method='usl_ccam')
    return model


def export(method,stage,manifest_path,out,config,gate_path,*,revision,sink=None):
    if stage not in {'generate','predict'}:raise ValueError('Seed generation or final prediction stage required')
    gate_report(gate_path,method,manifest_path,training=False)
    document=read_manifest(manifest_path)
    out=Path(out)
    if out.exists():raise ValueError('Never overwrite another run')
    size=config.get('resolution',224);device=prepare_device(config);model=None
    if method=='sled':
        from ..sled.pipeline import SLEDConfig,segment
        sled_config=SLEDConfig(**config.get('sled',{}))
    elif stage=='generate' and method=='usl_net':model=_usl_generator(config)
    elif stage=='generate' and method=='rpi_net':
        from ..rpi_net.seeds import RPISeedGenerator
        bank=source_bank(config['source_bank'],('mocov3','mocov2','dino'))
        model=RPISeedGenerator(bank['mocov3'],bank['mocov2'],bank['dino'])
    elif stage=='generate':raise ValueError('TriDermNet consumes reviewed pseudo labels; it is not an independent seed generator')
    else:
        model=model_for(method)
        load_reviewed(model,config['checkpoint'],config['checkpoint_provenance'],expected_method=method)
    if model is not None:model.to(device).eval()
    out.mkdir(parents=True);records=[];failures=0
    origin={'method':method,'producer_revision':revision,'uses_expert_masks':False,'uses_diagnoses':False}
    for index,row in enumerate(document['records']):
        started=time.monotonic();entry={k:row[k] for k in ('image_id','dataset','image_path','image_sha256')}
        case=out/f'case_{index:06d}';case.mkdir()
        try:
            rgb=rgb_tensor_array(row,size);valid=valid_array(row,size)
            if method=='sled':
                result=segment(np.rint(rgb.transpose(1,2,0)*255).astype(np.uint8),valid[0].astype(bool),sled_config)
                maps=result['mask'][None].astype(np.float32);audit=result['metadata']
            else:
                images=torch.from_numpy(rgb)[None].to(device)
                if stage=='generate' and method=='rpi_net':
                    with torch.enable_grad():maps=model(images,config.get('seed',42))[0]
                    audit={'sources':4,'cam_target':'ssl_feature_energy','cluster_polarity':'border_majority_background'}
                else:
                    with torch.no_grad():output=model(images)
                    if stage=='generate':
                        value=output[0,0].cpu().numpy()
                        pseudo,audit=usl_text_pseudolabel(value,USLTextPolicy(**config.get('usl_policy',{})));maps=pseudo[None]
                    else:
                        probability=output['logits'].sigmoid()[0,0].cpu().numpy()
                        np.save(case/'probability.npy',probability)
                        maps=(probability>=.5)[None].astype(np.float32);audit={'threshold':.5,'output':'final_mask_not_confident_pixels_only'}
            maps[:,valid[0]==0]=.5
            from .review import save_review
            save_review(rgb,maps,valid[0],case)
            paths=[];hashes=[]
            for source,value in enumerate(maps):
                path=case/f'pseudo_{source}.npy';np.save(path,value)
                paths.append(str(path.resolve()));hashes.append(digest(path))
            np.save(case/'valid.npy',valid[0]);valid_path=case/'valid.npy'
            entry.update(pseudo_paths=paths,pseudo_sha256s=hashes,pseudo_provenance=[origin.copy() for _ in paths],
                         valid_path=str(valid_path.resolve()),valid_sha256=digest(valid_path))
            record={'index':index,'status':'ok','elapsed_seconds':time.monotonic()-started,
                    'foreground_fraction':float((maps>=.65).mean()),'audit':audit}
            records.append(entry)
        except DegenerateCase as error:
            failures+=1;record={'index':index,'status':'abstained','reason':str(error),'elapsed_seconds':time.monotonic()-started}
        # Geometry/hash/runtime failures stop the run; do not hide them as abstention.
        with (out/'cases.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        if sink is not None:sink.log({'processed':index+1,'abstentions':failures,'case_seconds':record['elapsed_seconds']})
    # Keep original denominator in evidence. Refuse to call a partial list the full pool.
    report={'method':method,'stage':stage,'input_count':len(document['records']),'successful_count':len(records),'abstentions':failures,
            'input_manifest_sha256':digest(manifest_path),'producer_revision':revision}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    document['records']=records;document['count']=len(records)
    (out/'successful_pseudos_manifest.json').write_text(json.dumps(document,indent=2)+'\n')
    if failures:raise ValueError('Abstentions retained in cases.jsonl; partial pool needs explicit downstream policy, no silent filtering')
    return report
