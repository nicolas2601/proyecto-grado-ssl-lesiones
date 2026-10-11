"""Build a final index without rewriting preregistered or stage reports."""
import argparse, html, json, shutil
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    x = p.parse_args()
    w = x.root / 'w01_w2_s42_20261011'
    e = x.root / 'w2_eval_isic2018_s44_20261011'
    s = json.loads((e / 'summary.json').read_text())
    check = json.loads((e / 'verification.json').read_text())
    assert s['status'] == 'complete' and check['status'] == 'passed'
    source = Path(__file__).resolve().parents[1] / 'docs/39-resultados-y-puertas-v8.md'
    shutil.copy2(source, w / 'DECISIONS.md')
    text = '<html><meta charset="utf-8"><title>Auditoría v8 — resultados y decisiones</title><style>body{font-family:Georgia;color:#402c20;max-width:1050px;margin:36px auto;line-height:1.5;background:white}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #bcaa99;text-align:left}a{color:#6f4628}pre{white-space:pre-wrap}</style>'
    text += '<h1>Auditoría modular v8: resultados verificados</h1><p>W0/W1/W2 ejecutados en copia separada. Cambios pendientes del clúster conservados.</p>'
    text += '<p><b>Pool inicial 58.457 → elegibles 55.034 → SSL efectivo 40.896.</b><br>Cuarentena: 3.423; exclusiones históricas: 14.128; duplicados adicionales: 10.</p>'
    text += '<h2>100 ternas reales + 100 casos de cuarentena</h2><p>Rutas, índices y máscaras verificados. Un recorte sin pseudolesión y otro con solo 2,58% de retención. ITA disponible: I0 53/100, I1 e I2 57/100; no demuestra exactitud de color ni fototipo.</p>'
    text += '<h2>Semillas: contraste exploratorio con referencia experta</h2><p>100 pares ISIC 2018, semilla 44; todos coinciden por ID con el pool del maestro. Benchmark previamente consultado: <b>no evaluación externa independiente</b>. Dice incluye abstenciones; resultados de semillas, no del estudiante.</p>'
    text += '<table><tr><th>Brazo</th><th>Cambio</th><th>Accuracy px</th><th>Dice medio</th><th>Vacías /100</th><th>Δ Dice vs S0 [IC 95%]</th></tr>'
    names = {'S0':'Control','S1':'Abstención de mapa débil','S2':'Sin inversión centro/borde','S3':'Excluir pelo','S4':'Inpainting antes del maestro'}
    for arm, values in s['arms'].items():
        delta = s['paired_dice'].get(arm + '_minus_S0')
        label = '—' if delta is None else f"{delta['mean']:+.4f} [{delta['ci95_percentile'][0]:+.4f}; {delta['ci95_percentile'][1]:+.4f}]"
        text += f"<tr><td>{arm}</td><td>{names[arm]}</td><td>{check['mean_pixel_accuracy'][arm]:.4f}</td><td>{values['mean_dice']:.4f}</td><td>{values['empty_predictions']}</td><td>{label}</td></tr>"
    text += '</table><p>Accuracy cuenta también el gran fondo sano; por eso puede ser alta aunque el Dice y la sensibilidad de lesión sean modestos. IGNORE se contabiliza como no lesión en todo el marco.</p><p><b>Decisión:</b> conservar S0; no promover los pilotos a entrenamiento. S2 empeora claramente; S3 reduce ligeramente el Dice; S4 tiene mejora pequeña con intervalo que incluye cero. Cuarentena permanece excluida.</p>'
    text += '<p><b>Verificación:</b> 8 pruebas W0 + 5 pruebas de pilotos aprobadas; 200 IDs pareados, 800 máscaras W2 y 1400 archivos S4 verificados. Evaluación: 500 máscaras con hashes, métricas binarias y bootstrap recalculados.</p>'
    text += '<h2>Siguientes etapas</h2><p>W3 SLED: entorno separado pendiente (faltan Java/JPype/PyOD; Python actual 3.13). W3 espectral y W4 tono/dominio diseñados, sin nuevos entrenamientos. Próxima puerta: maestros independientes y evaluación fuera del pool comprobada por ID y contenido.</p>'
    text += '<p><a href="DECISIONS.md">Registro completo de decisiones</a> · <a href="summary.html">Pilotos W0/W1/W2 y 20 láminas</a> · <a href="../w2_eval_isic2018_s44_20261011/summary.html">Evaluación experta y 10 láminas</a></p>'
    text += '<details><summary>Detalles y límites</summary><pre>' + html.escape(source.read_text()) + '</pre></details></html>'
    (w / 'progress.html').write_text(text)
    print('FINAL INDEX COMPLETE')

if __name__ == '__main__':
    main()
