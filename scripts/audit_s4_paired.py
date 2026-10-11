from pathlib import Path
import sys,json,csv,hashlib,argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'scripts'),str(R/'src'),str(R/'plan-segmentacion/comun/nucleo'),str(R/'plan-segmentacion/camino-b-cnn/pseudo-mascaras'),str(R/'plan-segmentacion/camino-b-cnn/preentrenamiento-contrastivo')]
import audit_modular_followup as a
from contrastive_backbone import ResNet50MoCoFeatures
from ccam_engine import CCAMHead
from ckpt_io import load_checkpoint
import gpu_augment as ga

def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True);x=p.parse_args();out=x.out;out.mkdir(parents=True,exist_ok=False)
 assert torch.cuda.is_available(),'CUDA unavailable: S4 not run'
 base=x.project/'salidas/final';cfg=json.loads((base/'ccam/run_config.json').read_text());kwargs=json.loads((base/'pseudo_r0/pseudo_masks_stats.json').read_text());kwargs.setdefault('region_mode','central')
 ids=json.loads((x.previous/'selection.json').read_text());images=np.load(base/'cache/images.npy',mmap_mode='r');fields=np.load(base/'cache/field.npy',mmap_mode='r');old=np.load(base/'ccam/saliency.npy',mmap_mode='r')
 backbone=Path(cfg['backbone']);headfile=base/'ccam/last.pt'
 a.write(out/'preregistration.json',{'protocol':(R/'docs/37-s4-inferencia-pareada.md').read_text(),'protocol_sha':a.sha(R/'docs/37-s4-inferencia-pareada.md'),'head_sha256':a.sha(headfile),'backbone_sha256':a.sha(backbone),'selection_sha256':a.sha(x.previous/'selection.json'),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'precision':cfg['precision'],'paired_batch_size':8})
 provider=ResNet50MoCoFeatures.from_checkpoint(backbone).cuda().eval();head=CCAMHead(provider.out_channels,cfg['head_kernel']).cuda().eval();head.load_state_dict(load_checkpoint(headfile)['head'])
 records=[];gate_failed=False
 for start in range(0,len(ids),4):
  chunk=ids[start:start+4];rgbs=[];fractions=[]
  for r in chunk:
   rgb=np.asarray(images[r['native_index']]);clean,fraction=a.hr.remove_hair(rgb,shaped=True);rgbs.extend([rgb,clean]);fractions.append(fraction)
  batch=ga.normalize_batch(ga.to_float_batch(torch.from_numpy(np.stack(rgbs)).cuda()))
  with torch.no_grad():
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg['precision']=='bf16'):features=provider.extract(batch).float()
   pred=F.interpolate(head(features),size=(224,224),mode='bilinear',align_corners=False).squeeze(1).cpu().numpy().astype(np.float16).astype(np.float32)
  for i,r in enumerate(chunk):
   idx=r['native_index'];rgb=rgbs[2*i];field=np.asarray(fields[idx]).astype(bool);raw0,raw4=pred[2*i:2*i+2];delta=np.abs(raw0-np.asarray(old[idx],np.float32));pass_gate=float(delta.mean())<=.005 and float(delta.max())<=.03;gate_failed|=not pass_gate
   folder=out/r['cohort']/r['image_id'];folder.mkdir(parents=True)
   seed0=a.pilot(raw0,field,rgb,kwargs,'S0');seed4=a.pilot(raw4,field,rgb,kwargs,'S0')
   for name,result,raw in [('S0_reinferred',seed0,raw0),('S4',seed4,raw4)]:
    Image.fromarray(result.mask).save(folder/(name+'_labels.png'));a.overlay(rgb,result.mask==1,(170,95,45)).save(folder/(name+'_overlay.png'));np.save(folder/(name+'_raw.npy'),raw.astype(np.float16))
   Image.fromarray(rgbs[2*i+1]).save(folder/'inpainted.png')
   rec={**r,'hair_inpaint_fraction':fractions[i],'historical_mae':float(delta.mean()),'historical_max':float(delta.max()),'numerical_gate_pass':pass_gate,'paired_raw_mae':float(np.abs(raw4-raw0).mean()),'S0_valid':seed0.valid,'S4_valid':seed4.valid,'S4_reason':seed4.reason,'seed_changed_vs_paired_S0':not np.array_equal(seed0.mask,seed4.mask),'paired_label_changed_fraction':float((seed0.mask!=seed4.mask).mean()),'files_sha256':{p.name:a.sha(p) for p in folder.iterdir()}}
   records.append(rec)
  print('S4 PROGRESS',min(start+4,len(ids)),len(ids),flush=True)
 with (out/'records.jsonl').open('w') as f:
  for r in records:f.write(json.dumps(r)+'\n')
 summary={'status':'numerical_gate_failed' if gate_failed else 'complete','n':len(records),'numerical_gate_failed':sum(not r['numerical_gate_pass'] for r in records),'cohorts':{c:{'n':len(rs:=[r for r in records if r['cohort']==c]),'S0_valid':sum(r['S0_valid'] for r in rs),'S4_valid':sum(r['S4_valid'] for r in rs),'seeds_changed':sum(r['seed_changed_vs_paired_S0'] for r in rs),'mean_paired_raw_mae':float(np.mean([r['paired_raw_mae'] for r in rs])),'mean_inpaint_fraction':float(np.mean([r['hair_inpaint_fraction'] for r in rs]))} for c in ['training100','quarantine100']},'historical_mae_max':max(r['historical_mae'] for r in records),'historical_pixel_error_max':max(r['historical_max'] for r in records),'scope':'Same frozen teacher; input intervention only. No student training, anatomical evaluation or ITA on synthesized color.'}
 a.write(out/'summary.json',summary);a.write(out/'status.json',{'status':summary['status'],'n':len(records)});print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
