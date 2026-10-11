"""Build/check the numeric report and verify persisted masks. No data selection."""
from pathlib import Path
import sys,json,argparse,shutil,tempfile
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).parent))
import audit_modular_followup as a

def text_report(s,t,e):
 text='# Auditoría modular v8: resultados de desarrollo\n\n'
 text+='Ejecutado en copia aislada. Pool inicial **58.457**; exportación elegible **55.034**; cuarentena **3.423**; pool SSL histórico **40.896** después de exclusiones y deduplicación.\n\n'
 text+='100 IDs de entrenamiento (W0, semilla 42) + diagnóstico separado de 100 IDs de cuarentena (25 por dataset, semilla 43). Todos pareados contra el manifiesto/cache nativo. Sin diagnósticos, máscaras expertas ni PAD target_test.\n\n'
 text+='## W0\n\nUn recorte v2 sin pseudolesión. Otro v1 conserva solo 2,58% de la pseudolesión: la ausencia total no captura todos los extremos. Desgloses por dataset, área y artefactos en JSON. No estimar prevalencia a partir de estos estratos pequeños.\n\n'
 text+='## W1\n\n| Región | ITA disponible /100 | Indeterminado | Anillo insuficiente |\n|---|---:|---:|---:|\n'
 for arm in ['I0','I1','I2']:
  c=s['W1'][arm]['status_counts'];text+=f"| {arm} | {c.get('ok',0)} | {c.get('ita_indeterminate',0)} | {c.get('ring_too_small',0)} |\n"
 p=s['W1']['I1_I2_paired'];text+=f"\nI1/I2: {p['n_both_available']} pares disponibles; {p['bucket_changed']} categorías ITA cambiadas; cambio máximo de ITA {p['max_abs_ita_change']:.3f}°. Sin referencia de color/experta, no exactitud ni Fitzpatrick. Comparar medias de todos los disponibles mezcla subconjuntos; contraste pareado separado en JSON.\n\n"
 text+='## W2\n\n| Piloto | Semillas aceptadas: entrenamiento /100 | Cuarentena /100 |\n|---|---:|---:|\n'
 for arm in ['S0','S1','S2','S3']:
  text+=f"| {arm} | {s['W2']['training100']['arms'][arm]['valid']} | {s['W2']['quarantine100']['arms'][arm]['valid']} |\n"
 text+=f"| S4 | {t['cohorts']['training100']['S4_valid']} | {t['cohorts']['quarantine100']['S4_valid']} |\n"
 text+='\nS0 reproduce exactamente los 200 slices de pseudo_r0. S1 (span original ≤0,01) no cambia ninguno: esta muestra no aporta evidencia favorable para esa abstención. S2 sin inversión cambia muchos mapas y reduce aceptación en entrenamiento; aceptación no prueba anatomía. S3 cambia 62/100 mapas de entrenamiento, sin cambiar su aceptación; queda candidato a evaluación de contaminación por pelo. S4 usa el mismo maestro y presupuesto de inferencia, con/sin inpainting, no otro entrenamiento.\n\n'
 text+=f"S4 superó la puerta numérica en {t['n']}/{t['n']} casos; MAE histórica máxima por mapa {t['historical_mae_max']:.6f}. Cambió {t['cohorts']['training100']['seeds_changed']}/100 semillas de entrenamiento; {t['cohorts']['quarantine100']['seeds_changed']}/100 de cuarentena. No medir ITA en color sintetizado.\n\n"
 text+='## Decisiones\n\nConservar S0 como control; no adoptar S1/S2/S3/S4 por aceptación o acuerdo con r2. No reintroducir cuarentena. Priorizar evaluación anatómica independiente de S0/S3/S4 con criterios congelados y muestras separadas, y piloto de maestro independiente. Sin referencia experta no se reportan Dice/IoU anatómicos, HD95 ni superioridad. No se lanzó nuevo estudiante ni preentrenamiento SSL.\n\n'
 text+='W3 (SLED/espectral) y W4 (factorial de tono/dominio) permanecen diseñados detrás de esa evaluación. No hay resultados nuevos de esas fases. Código de ejecución basado en fuente f8b1790; checkout local conserva base 9c23daf con snapshot completo de src y entradas verificadas, no un checkout completo de f8b1790. Resultados históricos y cambios pendientes originales preservados.\n'
 return text

def main():
 p=argparse.ArgumentParser();p.add_argument('--w01',type=Path,required=True);p.add_argument('--s4',type=Path,required=True);p.add_argument('--check',action='store_true');x=p.parse_args();root=x.w01
 rows=[json.loads(v) for v in (root/'records.jsonl').read_text().splitlines()];s=json.loads((root/'summary.json').read_text());t=json.loads((x.s4/'summary.json').read_text());sr=[json.loads(v) for v in (x.s4/'records.jsonl').read_text().splitlines()]
 assert len(rows)==len(sr)==200 and [(r['image_id'],r['cohort']) for r in rows]==[(r['image_id'],r['cohort']) for r in sr]
 for r,u in zip(rows,sr):
  folder=root/r['cohort']/r['image_id']
  for arm in ['S0','S1','S2','S3']:
   mask=np.asarray(Image.open(folder/(arm+'_labels.png')));assert set(np.unique(mask))<={0,1,255};assert a.array_sha(mask)==r['seeds'][arm]['mask_sha256']
  for file,h in u['files_sha256'].items():assert a.sha(x.s4/r['cohort']/r['image_id']/file)==h
 train=[r for r in rows if r['cohort']=='training100'];extra={'W0_by_hair_flag':a.groups([{**r,'hair_flag':'detected_ge_0.005' if r['hair_fraction']>=.005 else 'below_0.005'} for r in train],'hair_flag'),'W0_by_field_flag':a.groups([{**r,'field_flag':'field_below_0.95' if r['field_fraction']<.95 else 'field_ge_0.95'} for r in train],'field_flag'),'extreme_retention':sorted([{'image_id':r['image_id'],'dataset':r['dataset'],'view':v['name'],'retained':v['lesion_retained_fraction']} for r in train for v in r['views']],key=lambda r:r['retained'])[:12],'quarantine_area_reasons':dict(Counter('lt_0.01' if r['mask_area']<.01 else 'gt_0.90' if r['mask_area']>.90 else 'within_area_bounds' for r in rows if r['cohort']=='quarantine100')),'paired_ITA':{}}
 for first,second in [('I0','I1'),('I0','I2')]:
  paired=[r for r in train if r[first]['status']==r[second]['status']=='ok'];d=[abs(r[first]['ita']-r[second]['ita']) for r in paired]
  extra['paired_ITA'][first+'_'+second]={'n':len(paired),'median_absolute_change':float(np.median(d)),'max_absolute_change':max(d),'categories_changed':sum(r[first]['bucket']!=r[second]['bucket'] for r in paired)}
 md=text_report(s,t,extra)
 if x.check:
  assert (root/'EXECUTION.md').read_text()==md,'report stale'
  assert json.loads((root/'extended_summary.json').read_text())==extra,'extended summary stale'
  print('CHECK PASSED: report, 200 paired IDs, 800 seed masks, 1400 S4 files',flush=True);return
 a.write(root/'extended_summary.json',extra);(root/'EXECUTION.md').write_text(md)
 # Direct sheet links avoid Jupyter HTML sandbox's authenticated image subresource limitation.
 for page in range(20):
  sheet=Image.new('RGB',(1344,2536),'white');draw=ImageDraw.Draw(sheet)
  for j,name in enumerate(['Original','Student r2','S0','S2 no inversion','S3 hair exclusion','S4 teacher + inpaint']):draw.text((j*224+4,4),name,fill='black')
  for i,r in enumerate(rows[page*10:(page+1)*10]):
   y=24+i*251;draw.text((4,y),r['cohort']+' '+r['image_id'],fill='black');f=root/r['cohort']/r['image_id'];sf=x.s4/r['cohort']/r['image_id']
   panel=Image.open(f/'panel.jpg');images=[panel.crop((j*224,32,(j+1)*224,256)) for j in [0,1,5,7,8]]+[Image.open(sf/'S4_overlay.png')]
   for j,im in enumerate(images):sheet.paste(im,(j*224,y+22))
  sheet.save(root/f'seeds_{page+1:02d}.jpg',quality=93)
 import html
 body='<html><meta charset="utf-8"><title>Auditoría v8 — W0 W1 W2 ejecutados</title><style>body{font-family:Georgia;color:#3c281c;max-width:1100px;margin:40px auto;background:white;line-height:1.5}table{border-collapse:collapse}td,th{padding:9px 16px;border-bottom:1px solid #baa48e}pre{white-space:pre-wrap}a{color:#714b2d}</style><h1>Auditoría modular v8: pilotos reales</h1><p><b>Verificado:</b> 100 casos de entrenamiento + 100 de cuarentena. W0 desglosado, I0/I1/I2 y S0–S4 ejecutados. Sin nuevo entrenamiento; W3/W4 diseñados.</p><p>Pool inicial: <b>58.457</b> · Exportado: <b>55.034</b> · Pool SSL: <b>40.896</b></p><table><tr><th>ITA</th><th>Disponible /100</th><th>Semillas</th><th>Entrenamiento /100</th><th>Cuarentena /100</th></tr>'
 for i,arm in enumerate(['S0','S1','S2','S3','S4']):
  ita=['I0','I1','I2'][i] if i<3 else '';available=s['W1'][ita]['status_counts']['ok'] if ita else '';v=s['W2']['training100']['arms'][arm]['valid'] if arm!='S4' else t['cohorts']['training100']['S4_valid'];q=s['W2']['quarantine100']['arms'][arm]['valid'] if arm!='S4' else t['cohorts']['quarantine100']['S4_valid'];body+=f'<tr><td>{ita}</td><td>{available}</td><td>{arm}</td><td>{v}</td><td>{q}</td></tr>'
 body+='</table><p>Más semillas aceptadas no significa mejor anatomía. ITA descriptivo, sin exactitud ni fototipo. Cuarentena permanece excluida.</p><p><a href="EXECUTION.md">Resultados, decisiones y límites</a> · <a href="summary.json">JSON W0/W1/W2</a> · <a href="extended_summary.json">Estratos y extremos</a> · <a href="records.jsonl">Registros por caso</a></p><h2>Inspección de originales y máscaras</h2><ul>'
 for page in range(20):body+=f'<li><a href="seeds_{page+1:02d}.jpg">Lámina {page+1}: {"entrenamiento" if page<10 else "cuarentena"} — casos {(page%10)*10+1}–{(page%10+1)*10}</a></li>'
 body+='</ul><details><summary>Reporte completo</summary><pre>'+html.escape(md)+'</pre></details></html>'
 (root/'summary.html').write_text(body)
 dest=Path(__file__).resolve().parents[1]/'resultados/auditoria_v8'/root.name
 for name in ['summary.json','extended_summary.json','EXECUTION.md']:shutil.copy2(root/name,dest/name)
 sd=Path(__file__).resolve().parents[1]/'resultados/auditoria_v8'/x.s4.name;sd.mkdir(parents=True,exist_ok=False)
 for name in ['summary.json','preregistration.json','records.jsonl','status.json']:shutil.copy2(x.s4/name,sd/name)
 print(json.dumps(extra,indent=2),flush=True)
if __name__=='__main__':main()
