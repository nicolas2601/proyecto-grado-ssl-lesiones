"""Frozen seed pilots scored separately; references opened only by eval_inputs."""
from pathlib import Path
import argparse,sys,json,csv,shutil
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'scripts'),str(R/'src'),str(R/'plan-segmentacion/comun/nucleo'),str(R/'plan-segmentacion/comun/evaluacion'),str(R/'plan-segmentacion/camino-b-cnn/pseudo-mascaras'),str(R/'plan-segmentacion/camino-b-cnn/preentrenamiento-contrastivo')]
import audit_modular_followup as a
import eval_inputs as ei,evaluate_segmentation as ev,field_roi as field
from contrastive_backbone import ResNet50MoCoFeatures
from ccam_engine import CCAMHead
from ckpt_io import load_checkpoint
import gpu_augment as ga
ARMS=['S0','S1','S2','S3','S4']
def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--image-dir',type=Path,required=True);p.add_argument('--mask-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);x=p.parse_args();o=x.out;o.mkdir(parents=True,exist_ok=False);a.write(o/'status.json',{'status':'predicting'})
 pairs=ei.pair_images_and_masks(x.image_dir,x.mask_dir);assert len(pairs)>=100
 rng=np.random.default_rng(44);selected=[pairs[i] for i in sorted(rng.choice(len(pairs),size=100,replace=False).tolist())]
 base=x.project/'salidas/final';cfg=json.loads((base/'ccam/run_config.json').read_text());kwargs=json.loads((base/'pseudo_r0/pseudo_masks_stats.json').read_text());kwargs.setdefault('region_mode','central')
 native=list(csv.DictReader((base/'cache/manifest.csv').open()));canon={r['image_id'].removesuffix('_downsampled') for r in native}
 selection=[{'image_id':i,'image_path':str(ip),'mask_path':str(mp),'image_sha256':a.sha(ip),'mask_sha256':a.sha(mp),'in_pool_by_canonical_id':i in canon} for i,ip,mp in selected]
 a.write(o/'selection.json',selection);shutil.copy2(R/'docs/38-evaluacion-semillas-v8.md',o/'protocol.md')
 a.write(o/'provenance.json',{'seed':44,'bootstrap_seed':45,'bootstrap_n':4000,'source_pairs':len(pairs),'selected_n':100,'selected_in_pool_by_id':sum(r['in_pool_by_canonical_id'] for r in selection),'scope':'Exploratory previously evaluated benchmark; pool overlap by canonical ID, not content-proof; never an intact external test','head_sha256':a.sha(base/'ccam/last.pt'),'backbone_sha256':a.sha(Path(cfg['backbone'])),'config':kwargs,'protocol_sha256':a.sha(o/'protocol.md'),'script_sha256':a.sha(Path(__file__)),'ignore_evaluation':'foreground == 1, IGNORE as no lesion, full grid scored','torch':torch.__version__})
 assert torch.cuda.is_available();provider=ResNet50MoCoFeatures.from_checkpoint(Path(cfg['backbone'])).cuda().eval();head=CCAMHead(provider.out_channels,cfg['head_kernel']).cuda().eval();head.load_state_dict(load_checkpoint(base/'ccam/last.pt')['head'])
 for start in range(0,100,4):
  chunk=selected[start:start+4];rgbs=[];fields=[]
  for iid,ip,mp in chunk:
   rgb=ei.load_image(ip,224);clean,fraction=a.hr.remove_hair(rgb,shaped=True);rgbs.extend([rgb,clean]);fields.append(field.field_validity_mask(rgb)[0])
  batch=ga.normalize_batch(ga.to_float_batch(torch.from_numpy(np.stack(rgbs)).cuda()))
  with torch.no_grad():
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg['precision']=='bf16'):features=provider.extract(batch).float()
   pred=F.interpolate(head(features),size=(224,224),mode='bilinear',align_corners=False).squeeze(1).cpu().numpy().astype(np.float16).astype(np.float32)
  for i,(iid,ip,mp) in enumerate(chunk):
   folder=o/iid;folder.mkdir();rgb=rgbs[2*i];fld=fields[i]
   Image.fromarray(rgb).save(folder/'original.png');Image.fromarray(rgbs[2*i+1]).save(folder/'inpainted.png')
   for arm in ARMS:
    raw=pred[2*i+(arm=='S4')];result=a.pilot(raw,fld,rgb,kwargs,'S0' if arm=='S4' else arm)
    Image.fromarray(result.mask).save(folder/(arm+'_labels.png'));a.overlay(rgb,result.mask==1,(170,95,45)).save(folder/(arm+'_overlay.png'))
   np.save(folder/'raw_original.npy',pred[2*i].astype(np.float16));np.save(folder/'raw_inpainted.npy',pred[2*i+1].astype(np.float16))
  print('PREDICTED',min(start+4,100),flush=True)
 # All predictions complete before reference masks are decoded. Never used by teacher.
 a.write(o/'status.json',{'status':'evaluating','predictions_complete':100})
 rows=[]
 for iid,ip,mp in selected:
  truth=ei.load_binary_mask(mp,224);folder=o/iid;Image.fromarray((truth*255).astype(np.uint8)).save(folder/'reference.png');row={'image_id':iid,'arms':{}}
  for arm in ARMS:
   labels=np.asarray(Image.open(folder/(arm+'_labels.png')));pred=labels==1
   row['arms'][arm]={**ev.compute_binary_metrics(pred,truth),'hd95':ev.hausdorff95(pred,truth),'empty_prediction':not pred.any(),'ignore_fraction':float((labels==255).mean()),'mask_sha256':a.sha(folder/(arm+'_labels.png'))}
  rows.append(row)
 with (o/'records.jsonl').open('w') as f:
  for r in rows:f.write(json.dumps(a.clean(r),default=lambda v:v.item(),allow_nan=False)+'\n')
 summary={'status':'complete','n':100,'exploratory':True,'n_in_pool_by_canonical_id':sum(r['in_pool_by_canonical_id'] for r in selection),'arms':{},'paired_dice':{}}
 for arm in ARMS:
  rs=[r['arms'][arm] for r in rows];summary['arms'][arm]={'n':100,'empty_predictions':sum(r['empty_prediction'] for r in rs),'mean_dice':float(np.mean([r['dice'] for r in rs])),'p10_dice':float(np.quantile([r['dice'] for r in rs],.1)),'mean_iou':float(np.mean([r['iou'] for r in rs])),'mean_sensitivity':float(np.nanmean([r['sensitivity'] for r in rs])),'mean_hd95_defined':float(np.nanmean([r['hd95'] for r in rs])),'n_hd95_defined':int(sum(np.isfinite(r['hd95']) for r in rs))}
 rng=np.random.default_rng(45);draws=rng.integers(0,100,size=(4000,100))
 for arm in ARMS[1:]:
  diff=np.array([r['arms'][arm]['dice']-r['arms']['S0']['dice'] for r in rows]);ci=np.quantile(diff[draws].mean(axis=1),[.025,.975]);summary['paired_dice'][arm+'_minus_S0']={'mean':float(diff.mean()),'ci95_percentile':ci.tolist(),'better':int((diff>1e-12).sum()),'worse':int((diff< -1e-12).sum()),'tied':int((abs(diff)<=1e-12).sum())}
 summary['limits']=['Already consulted benchmark, exploratory not confirmation. Pool overlap by ID reported.','Seed predictions only, not student segmentation; no new training.','All100 included in Dice/IoU; HD95 undefined cases explicitly counted.','Do not tune S1 threshold, orientation, checkpoint or exclusion rules with this evaluation.']
 a.write(o/'summary.json',summary);a.write(o/'status.json',{'status':'complete','n':100})
 for page in range(10):
  sheet=Image.new('RGB',(1568,2528),'white');d=ImageDraw.Draw(sheet)
  for j,name in enumerate(['Original','Reference']+ARMS):d.text((224*j+4,4),name,fill='black')
  for pos,r in enumerate(rows[page*10:(page+1)*10]):
   y=24+pos*250;d.text((4,y),r['image_id']+' Dice '+','.join(f"{r['arms'][arm]['dice']:.2f}" for arm in ARMS),fill='black');f=o/r['image_id']
   images=[Image.open(f/'original.png'),a.overlay(np.asarray(Image.open(f/'original.png')),np.asarray(Image.open(f/'reference.png'))>0,(75,150,90))]+[Image.open(f/(arm+'_overlay.png')) for arm in ARMS]
   for j,im in enumerate(images):sheet.paste(im,(j*224,y+22))
  sheet.save(o/f'eval_{page+1:02d}.jpg',quality=93)
 import html
 (o/'summary.html').write_text('<html><meta charset="utf-8"><title>Evaluación anatómica exploratoria v8</title><h1>Semillas S0–S4: evaluación exploratoria</h1><p>Benchmark ISIC2018 previamente consultado; referencia experta solo para métricas. No nuevo entrenamiento.</p><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre><ul>'+''.join(f'<li><a href="eval_{i:02d}.jpg">Lámina {i}</a></li>' for i in range(1,11))+'</ul></html>')
 dest=R/'resultados/auditoria_v8'/o.name;dest.mkdir(parents=True,exist_ok=False)
 for name in ['summary.json','protocol.md','provenance.json','selection.json','records.jsonl','status.json']:shutil.copy2(o/name,dest/name)
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
