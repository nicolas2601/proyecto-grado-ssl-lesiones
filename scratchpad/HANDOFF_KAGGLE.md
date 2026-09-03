# Handoff: correr el banco de baselines en Kaggle (Objetivo 2, proyecto de grado UNAB)

Contexto para una sesion nueva. Todo lo de abajo esta verificado el 2026-09-03,
no es plan teorico. Repo: `~/Documentos/proyecto-grado-ssl-lesiones`, rama
`feature/v6-gpu-ssl-pretraining` (limpia, pusheada, 227 tests verdes).

## Objetivo

Ejecutar en Kaggle, por CLI y sin navegador, el banco de tests de backbones
congelados sobre dos datasets, recoger los JSON de resultados, commitearlos y
armar la tabla comparativa. Son 16 corridas.

## Credenciales (ya configuradas, no volver a pedirlas)

- `~/.kaggle/access_token` (chmod 600). Usuario Kaggle: `nicorghost`.
- CLI: `export PATH="$HOME/.local/bin:$PATH"` -> `kaggle` 2.2.4. Verificar con
  `kaggle kernels list --user nicorghost`.
- Alternativa sin token: `composio proxy <url> --toolkit kaggle`.
- SEGURIDAD: el token se expuso en un chat. Al terminar todas las corridas,
  recordarle a Nico rotarlo en https://www.kaggle.com/settings/api.

## Herramienta principal

`python scripts/kaggle_push.py` genera `kernel-metadata.json`, opcionalmente
pone `SMOKE = False`, y sube + ejecuta el notebook.

```bash
# humo (SMOKE=True, ~5 min): valida cableado, numeros NO validos
python scripts/kaggle_push.py --bank baselines_isic2020 --user nicorghost --models resnet50
# corrida real de todo el banco
python scripts/kaggle_push.py --bank baselines_isic2020 --user nicorghost --real
# estado y descarga
python scripts/kaggle_push.py --status --bank baselines_isic2020 --user nicorghost
kaggle kernels output nicorghost/baselines-isic2020-resnet50 -p results/baselines_isic2020/
```

Bancos disponibles: `baselines` (HAM10000), `baselines_isic2020`,
`finetune`, `finetune_isic2020`. Modelos: alexnet, vgg19, efficientnet_b0,
resnet50, vit_b16, simsiam_r50, dinov2_vits14, dinov2_vitb14.

## Trampas ya pagadas (no repetirlas)

1. **Nunca P100.** El PyTorch de Kaggle solo trae kernels sm_70..sm_120; una
   Tesla P100 (sm_60) revienta en el primer forward con
   `AcceleratorError: CUDA error: no kernel image is available`. Por eso
   `kaggle_push.py` fija `machine_shape: "NvidiaTeslaT4"`. El enum del SDK
   admite solo `NvidiaTeslaT4` y `NvidiaTeslaP100`; `NvidiaTeslaT4x2` no
   existe y Kaggle lo ignora en silencio.
2. **AUC binario.** Ya corregido (commit b540e84): con 2 clases sklearn quiere
   `roc_auc_score(y, proba[:,1])`, no `multi_class="ovr"` sobre la matriz.
3. **Leer por que fallo un kernel**:
   `composio proxy 'https://www.kaggle.com/api/v1/kernels/output?userName=nicorghost&kernelSlug=<slug>' --toolkit kaggle`
   y mirar el campo `log`.
4. **Cuota GPU Kaggle**: ~30 h/semana. Un banco congelado completo son ~2-3 h.

## Orden de trabajo

1. Humo de `baselines_isic2020` con un modelo (resnet50). Verificar que
   termina en `complete` y que el Output trae `results_resnet50_isic2020.json`.
2. Si pasa: `--real` para los 8 modelos de `baselines_isic2020`.
3. Re-correr `baselines` (HAM10000) con `--real` para los 8: los JSON viejos
   no traen precision/F1 por clase, matriz de confusion ni IC bootstrap 95%, y
   ademas `vit_b16` y `efficientnet_b0` cambiaron de pesos al migrar a Hugging
   Face. Misma semilla 42 reproduce los numeros de los otros seis.
4. Guardar los JSON en `results/baselines/` y `results/baselines_isic2020/`,
   commitear con conventional commits (sin co-author, sin rastro de IA) y
   pushear a `feature/v6-gpu-ssl-pretraining`.
5. Actualizar la tabla de resultados de `notebooks/baselines/README.md` y del
   `README.md` principal con los numeros nuevos.
6. Humos del banco de fine-tuning (`finetune`, `finetune_isic2020`) para
   validar el cableado. Las corridas reales de fine-tuning NO van aca: las
   ejecuta la Dra. Karen en su cluster (`cluster/finetune/run_all.sh`).

## Fuera de alcance

ISIC 2019 quedo fuera de los bancos de Kaggle (decision de Nico, 2026-09-03).
El dataset sigue usandose en el pool SSL y en validacion externa: no borrar
esas rutas de codigo.

## Definicion de terminado

Los 16 JSON en `results/`, commiteados y pusheados; las tablas de los README
actualizadas; y un resumen de que modelo gano en cada dataset, con la lectura
honesta de si el SSL generico (DINOv2) alcanza al supervisado a igual tamano.
