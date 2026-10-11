"""W0/W1 descriptors and W2 seed pilots; never changes historical outputs."""
from __future__ import annotations
import argparse,csv,json,hashlib,sys,shutil
from pathlib import Path
from collections import Counter,defaultdict
from dataclasses import asdict
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'plan-segmentacion/comun/nucleo'),str(REPO/'plan-segmentacion/camino-b-cnn/pseudo-mascaras')]
from dermssl import fitzpatrick as fz,e5_roi,e3_geometry as geo,mask_manifest
import hair_removal as hr, uncertainty_module as um,saliency_prep as sp
from artifacts import artifact_mask

def clean(obj):
 if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
 if isinstance(obj,(list,tuple)):return [clean(v) for v in obj]
 if isinstance(obj,np.generic):obj=obj.item()
 if isinstance(obj,float) and not np.isfinite(obj):return None
 return obj

def write(path,obj):path.write_text(json.dumps(clean(obj),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def array_sha(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()
def components(mask):
 lab,n=ndimage.label(mask,np.ones((3,3)));areas=np.bincount(lab.ravel())[1:]
 return {'n_components':int(n),'largest_fraction':float(areas.max()/mask.sum()) if n else None}
def pilot(raw,field,rgb,kwargs,arm):
 if not np.isfinite(raw).all():return um.PseudoMaskResult(np.full(raw.shape,255,np.uint8),False,'nonfinite')
 if arm=='S1' and float(np.ptp(raw.astype(np.float32)))<=0.01:
  return um.PseudoMaskResult(np.full(raw.shape,255,np.uint8),False,'weak_span_le_0.01')
 normalized=sp.normalize_saliency(raw)
 if arm!='S2':normalized=sp.orient_saliency(normalized,field=field)
 artifact=artifact_mask(rgb) if kwargs['artifact_suppression'] else None
 if arm=='S3':
  hair=hr.elongated_hair_mask(rgb).astype(bool);artifact=hair if artifact is None else artifact|hair
 params={k:kwargs[k] for k in ['low','high','min_area_fraction','max_center_distance','max_area_fraction','region_mode']}
 return um.saliency_to_pseudo_mask(normalized,normalize=False,field=field,artifact=artifact,**params)
def overlay(rgb,mask,color):
 out=rgb.copy();out[mask]=np.rint(.55*out[mask]+.45*np.array(color)).astype(np.uint8);return Image.fromarray(out)
def groups(records,key):
 out={}
 for v in sorted({r[key] for r in records}):
  rows=[r for r in records if r[key]==v];stats={}
  for view in ['v1_original','v2_transformed','tone_aligned']:
   vals=[next(t for t in r['views'] if t['name']==view) for r in rows]
   ret=[t['lesion_retained_fraction'] for t in vals]
   stats[view]={'n':len(vals),'absent':sum(t['lesion_absent'] for t in vals),'mean':float(np.mean(ret)),'p10':float(np.quantile(ret,.1)),'min':float(min(ret))}
  out[v]=stats
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--w0',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 out=args.out;out.mkdir(parents=True,exist_ok=False);write(out/'status.json',{'status':'running'})
 protocol=REPO/'docs/36-pilotos-v8-desarrollo.md';shutil.copy2(protocol,out/protocol.name)
 cases=[json.loads(s) for s in (args.w0/'cases.jsonl').read_text().splitlines()]
 assert len(cases)==100
 base=args.project/'salidas/final';inputs=args.project/'salidas/tone/inputs'
 native=list(csv.DictReader((base/'cache/manifest.csv').open()));byid={r['image_id']:r for r in native}
 assert len(byid)==len(native)==58457
 records=json.loads((inputs/'mask_manifest.json').read_text())['masks'];mby={r['image_id']:r for r in records};assert len(mby)==58457
 images=np.load(base/'cache/images.npy',mmap_mode='r');fields=np.load(base/'cache/field.npy',mmap_mode='r');sal=np.load(base/'ccam/saliency.npy',mmap_mode='r');pseudo=np.load(base/'pseudo_r2/pseudo_masks.npy',mmap_mode='r');p0=np.load(base/'pseudo_r0/pseudo_masks.npy',mmap_mode='r')
 assert images.shape==(58457,224,224,3) and fields.shape==sal.shape==pseudo.shape==p0.shape==(58457,224,224)
 cfg=json.loads((base/'pseudo_r0/pseudo_masks_stats.json').read_text());cfg.setdefault('region_mode','central')
 assert cfg['source']=='ccam' and cfg['field_used'] and cfg['images']==str(base/'cache/images.npy') and cfg['saliency']==str(base/'ccam/saliency.npy')
 provenance={'protocol_sha256':sha(protocol),'script_sha256':sha(Path(__file__)),'manifest_sha256':sha(base/'cache/manifest.csv'),'mask_manifest_sha256':sha(inputs/'mask_manifest.json'),'w0_cases_sha256':sha(args.w0/'cases.jsonl'),'seed':42,'quarantine_seed':43,'w2_config':cfg,'native_arrays':{str(p):{'size':p.stat().st_size,'shape':list(np.load(p,mmap_mode='r').shape),'dtype':str(np.load(p,mmap_mode='r').dtype)} for p in [base/'cache/images.npy',base/'cache/field.npy',base/'ccam/saliency.npy',base/'pseudo_r2/pseudo_masks.npy']},'limitation':'Large arrays fingerprinted by manifest plus selected slice hashes; not full-file hashes. Quarantine sample is diagnostic, includes IDs excluded from SSL, never reintroduced. No diagnoses or expert masks loaded.'}
 write(out/'provenance.json',provenance)
 q=defaultdict(list)
 for rec in records:
  if not rec['eligible']:q[rec['dataset']].append(rec['image_id'])
 rng=np.random.default_rng(43);qids=[]
 for ds in sorted(q):qids+=sorted(rng.choice(sorted(q[ds]),size=min(25,len(q[ds])),replace=False).tolist())
 selected=[(c['image_id'],'training100',c) for c in cases]+[(i,'quarantine100',None) for i in qids]
 write(out/'selection.json',[{'image_id':i,'cohort':cohort,'native_index':int(byid[i]['index']),'dataset':byid[i]['dataset'],'split':byid[i]['split']} for i,cohort,c in selected])
 rows=[];w0=[];mismatch=0
 for num,(iid,cohort,c) in enumerate(selected):
  idx=int(byid[iid]['index']);rgb=np.asarray(images[idx]);field=np.asarray(fields[idx]).astype(bool);raw=np.asarray(sal[idx],dtype=np.float32);mask=np.asarray(pseudo[idx])==1
  assert mask_manifest.mask_sha256(mask)==mby[iid]['mask_sha256']
  if c:
   assert sha(Path(c['inputs']['original']['path']))==c['inputs']['original']['sha256']
   assert np.array_equal(rgb,np.asarray(Image.open(c['inputs']['original']['path']).convert('RGB')))
   assert np.array_equal(mask,np.asarray(Image.open(c['inputs']['pseudo_mask']['path']))>0)
  hair=hr.elongated_hair_mask(rgb).astype(bool);ring=geo.ring_mask(mask);img=rgb.astype(np.float32)/255.;lab=fz.srgb_to_lab(img)
  i0=asdict(fz.estimate_fst(img,mask));i1=asdict(e5_roi.ring_estimate(img,mask));i2=asdict(e5_roi.ring_estimate(img,mask,hair_fn=lambda _:hair))
  if cohort.startswith('quarantine'):
   # Preserve quarantine: no ITA result from a fallback mask is admitted as I0.
   i0={'status':'mask_ineligible','bucket':None,'ita':None}
   i1={'status':'mask_ineligible','bucket':None,'ita':None};i2=dict(i1)
  area=float(mask.mean());size='lt_0.05' if area<.05 else '0.05_to_0.20' if area<.2 else '0.20_to_0.50' if area<.5 else 'ge_0.50'
  record={'image_id':iid,'cohort':cohort,'dataset':byid[iid]['dataset'],'native_index':idx,'mask_area':area,'size_bin':size,'eligible':mby[iid]['eligible'],'quarantine_reason':mby[iid].get('quarantine_reason'),'hair_fraction':float(hair.mean()),'hair_in_lesion_fraction':float((hair&mask).sum()/max(1,mask.sum())),'hair_in_ring_fraction':float((hair&ring).sum()/max(1,ring.sum())),'field_fraction':float(field.mean()),'lesion_outside_field_fraction':float((mask&~field).sum()/max(1,mask.sum())),**components(mask),'I0':i0,'I1':i1,'I2':i2,'raw_min':float(raw.min()),'raw_max':float(raw.max()),'raw_span':float(np.ptp(raw)),'raw_std':float(raw.std()),'slice_sha256':{k:array_sha(a) for k,a in [('rgb',rgb),('field',field),('saliency',raw),('student_mask',np.asarray(pseudo[idx]))]}}
  if c:
   record['views']=c['views'];w0.append(record)
  normalized=sp.normalize_saliency(raw);oriented=sp.orient_saliency(normalized,field=field)
  record['orientation_flipped']=not np.array_equal(normalized,oriented)
  record['seeds']={};folder=out/cohort/iid;folder.mkdir(parents=True)
  for arm in ['S0','S1','S2','S3']:
   result=pilot(raw,field,rgb,cfg,arm);fg=result.mask==1
   record['seeds'][arm]={'valid':result.valid,'reason':result.reason,'n_regions':result.n_regions,'area':float(fg.mean()),'ignore_fraction':float((result.mask==255).mean()),'mask_sha256':array_sha(result.mask),'agreement_dice_vs_student':float(2*(fg&mask).sum()/(fg.sum()+mask.sum())) if fg.sum()+mask.sum() else None}
   Image.fromarray(result.mask).save(folder/(arm+'_labels.png'))
   overlay(rgb,fg,(170,95,45)).save(folder/(arm+'_overlay.png'))
   if arm=='S0':
    assert np.array_equal(result.mask,np.asarray(p0[idx])),f'{iid}: S0 does not reproduce historical pseudo_r0'
  record['seeds']['S4']={'status':'requires_paired_teacher_inference','reason':'inpainting changes teacher input; cannot be emulated by postprocessing existing map'}
  valid1=ring&~e5_roi.exclusion_mask(img,lab);valid2=valid1&~hair
  # Masks represent actual regions read by each estimator only in eligible cohort.
  if not mby[iid]['eligible']:valid1[:]=False;valid2[:]=False
  panels=[Image.fromarray(rgb),overlay(rgb,mask,(120,65,40)),overlay(rgb,valid1,(70,145,95)),overlay(rgb,valid2,(70,145,95)),overlay(rgb,hair,(180,75,155))]
  panels += [Image.open(folder/(arm+'_overlay.png')) for arm in ['S0','S1','S2','S3']]
  grid=Image.new('RGB',(224*9,260),'white');d=ImageDraw.Draw(grid)
  for j,(title,panel) in enumerate(zip(['Original','Student r2','I1 ring','I2 ring - hair','Hair','S0 historical','S1 weak abstain','S2 no inversion','S3 exclude hair'],panels)):
   d.text((224*j+3,4),title,fill='black');grid.paste(panel,(224*j,32))
  grid.save(folder/'panel.jpg',quality=93)
  rows.append(record)
  if (num+1)%25==0:print('PROGRESS',num+1,'/',len(selected),flush=True)
 with (out/'records.jsonl').open('w') as f:
  for r in rows:f.write(json.dumps(clean(r),allow_nan=False)+'\n')
 summary={'status':'complete','n_training':100,'n_quarantine':len(qids),'pool_counts':{'initial':len(records),'eligible_export':sum(r['eligible'] for r in records),'quarantine':sum(not r['eligible'] for r in records)},'per_dataset':{ds:dict(Counter('eligible' if r['eligible'] else 'quarantine' for r in records if r['dataset']==ds)) for ds in sorted(q)},'W0_by_dataset':groups(w0,'dataset'),'W0_by_mask_size':groups(w0,'size_bin'),'W1':{},'W2':{}}
 for arm in ['I0','I1','I2']:
  rs=[r[arm] for r in w0];ok=[r for r in rs if r['status']=='ok'];summary['W1'][arm]={'status_counts':dict(Counter(r['status'] for r in rs)),'mean_ita_available':float(np.mean([r['ita'] for r in ok])) if ok else None,'median_pixels_available':float(np.median([r.get('n_px',r.get('n_skin_px',0)) for r in ok])) if ok else None}
 pairs=[r for r in w0 if r['I1']['status']==r['I2']['status']=='ok'];summary['W1']['I1_I2_paired']={'n_both_available':len(pairs),'bucket_changed':sum(r['I1']['bucket']!=r['I2']['bucket'] for r in pairs),'median_abs_ita_change':float(np.median([abs(r['I1']['ita']-r['I2']['ita']) for r in pairs])) if pairs else None,'max_abs_ita_change':float(max(abs(r['I1']['ita']-r['I2']['ita']) for r in pairs)) if pairs else None}
 for cohort in ['training100','quarantine100']:
  rs=[r for r in rows if r['cohort']==cohort];summary['W2'][cohort]={'n':len(rs),'orientation_flipped':sum(r['orientation_flipped'] for r in rs),'raw_span_min':min(r['raw_span'] for r in rs),'raw_span_p10':float(np.quantile([r['raw_span'] for r in rs],.1)),'arms':{arm:{'valid':sum(r['seeds'][arm]['valid'] for r in rs),'reasons':dict(Counter(r['seeds'][arm]['reason'] for r in rs)),'changed_vs_S0':sum(r['seeds'][arm]['mask_sha256']!=r['seeds']['S0']['mask_sha256'] for r in rs)} for arm in ['S0','S1','S2','S3']}}
 summary['W2']['S0_reproduced_historical_selected_slices']=len(rows)
 summary['warnings']=['No expert color or masks: no accuracy, Dice/IoU anatomy or HD95 conclusions. Agreement vs student is not anatomical evaluation.','Quarantine sample stratified 25 per dataset: not prevalence-weighted and not admitted to training.','S1 threshold is an uncalibrated development hypothesis, never a probability.','W3/W4 require gate decisions from independent evaluation; no student or SSL training started.']
 write(out/'summary.json',summary);write(out/'status.json',{'status':'complete','n':len(rows)});report(out,summary,rows)
 dest=REPO/'resultados/auditoria_v8'/out.name;dest.mkdir(parents=True,exist_ok=False)
 for name in ['summary.json','provenance.json','selection.json','records.jsonl','status.json','36-pilotos-v8-desarrollo.md','README.md']:shutil.copy2(out/name,dest/name)
 print(json.dumps(clean(summary),indent=2),flush=True);print('OUTPUT',out,flush=True)

def report(out,s,rows):
 md='# W0/W1/W2: diagnóstico real de desarrollo\n\n'+ 'Fuente de cifras: `summary.json` y `records.jsonl`. 100 imágenes del pool SSL y 100 de cuarentena estratificadas; no mezclarlas.\n\n'
 md+='| ITA | Disponible | Abstenciones |\n|---|---:|---|\n'
 for arm in ['I0','I1','I2']:
  c=s['W1'][arm]['status_counts'];md+=f"| {arm} | {c.get('ok',0)} | {json.dumps({k:v for k,v in c.items() if k!='ok'})} |\n"
 md+='\nNo hay referencia experta ni de color: diferencias descriptivas, no exactitud ni Fitzpatrick. S0 reproduce los mapas históricos seleccionados. S4 exige inferencia pareada del maestro. W3/W4 siguen detrás de la puerta de evaluación.\n'
 (out/'README.md').write_text(md)
 import html
 body='<html><meta charset="utf-8"><title>Auditoría modular v8: W0 W1 W2</title><style>body{font-family:Georgia;background:#fff;color:#3c281c;max-width:1400px;margin:auto;padding:24px}img{width:100%}pre{white-space:pre-wrap}details{margin:20px 0}</style><h1>Auditoría modular v8: W0 / W1 / W2</h1><p>100 casos de entrenamiento + 100 diagnósticos de cuarentena. Sin etiquetas expertas. ITA descriptivo; S0–S3 son pilotos de semillas, no estudiantes entrenados.</p><pre>'+html.escape(json.dumps(clean(s),indent=2,ensure_ascii=False))+'</pre>'
 for r in rows:
  body+=f'<details><summary>{r["cohort"]} · {r["image_id"]} · {r["dataset"]}</summary><img src="{r["cohort"]}/{r["image_id"]}/panel.jpg"></details>'
 (out/'summary.html').write_text(body+'</html>')
if __name__=='__main__':main()
