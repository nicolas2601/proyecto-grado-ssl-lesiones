# Calibracion de tone_consistency_weight

## Que es esto

El entrenador ya tiene el mecanismo de consistencia tonal cableado
(`src/dermssl/pretrain/torch_adaptation.py`, `_lever_objective`), controlado
por `--tone-consistency-weight`. Su valor por defecto es 0.0 (apagado). Las
corridas `ssl_overlay_A_s42/43/44` ya existentes en
`clf1_clinical_reference_20261024/runs/` lo usan en 0.0 y sirven de control
sin tono para el contraste pareado.

El paper de referencia (CIRCLe, Pakzad et al. 2022, eq. 1) declara
`L_total = L_cls + lambda*L_reg` con `lambda in [0,1]` pero nunca publica el
valor usado en el paper. El valor por defecto del repo oficial
(github.com/arezou-pakzad/CIRCLe, `main.py`, flag `--alpha`) es **0.1**,
verificado que SI se aplica dentro de `models/circle.py`:
`reg = self.alpha * F.mse_loss(z_new, z)`.

Ese 0.1 no se puede copiar directo: la perdida base de CIRCLe es entropia
cruzada sobre 114 clases (arranca en torno a ln(114) ~= 4.7); la nuestra es
SimSiam (coseno negativo, rango -1 a 0). Un mismo peso numerico pesa una
fraccion muy distinta de cada base. Se calibro por sondeo en vez de copiar
el numero.

## Metodo

Dos corridas de 50 pasos sobre el pool real (33.620 imagenes,
`data_revision f4fcee05...`, espejo transformado `tone_s42`,
`transformed_data_revision d2890115...`), mismo split, misma semilla (42),
unica variable el peso:

| corrida | peso | ssl_loss | total_loss | razon medida |
|---|---:|---:|---:|---:|
| w1_s42 | 1.000 | -0.021686 | -0.010525 | 51.5 % |
| w0194_s42 | 0.194 | -0.027432 | -0.024702 | **9.95 %** |

Razon = `abs((total_loss - ssl_loss) / ssl_loss)`, es decir que fraccion de
la perdida base aporta el termino de tono ya ponderado.

El peso 0.194 se deriva de la primera corrida (razon 51.5 % con peso 1.0)
extrapolando a una razon objetivo del 10 %: `0.194 = 0.1 / 0.515`. La segunda
corrida verifica que la relacion es lineal: 9.95 % medido contra 10.0 %
esperado, error de 0.05 puntos. Dos corridas, no una, es lo que distingue
una calibracion de una conjetura con un solo dato.

## Valor recomendado

```
tone_consistency_weight = 0.194
```

Interpretacion para el informe: "peso derivado por sondeo de 50 pasos sobre
el pool real, fijado para que el termino de consistencia tonal aporte el
10% de la perdida base, en la misma proporcion relativa con la que CIRCLe
(lambda=0.1) pesa su regularizacion frente a su propia perdida de
clasificacion. No se copia el 0.1 porque las bases tienen escalas
distintas (entropia cruzada de 114 clases contra SimSiam)."

## No verificado

- La razon se midio a paso ~0 (effective_lr 7.8e-6, 50 pasos). Puede derivar
  durante una corrida larga de 6.000 pasos. Volver a medir a mitad de
  camino en la corrida real antes de darla por definitiva.
- No se corrio una tercera semilla de calibracion. El valor se apoya en dos
  puntos que verifican linealidad, no en variacion entre semillas.

## Reproducir

```
ssh unabia
cd ~/work/proyectonp/repo_v8_canonical
export GIT_COMMIT=$(git rev-parse HEAD)
PYTHONPATH=src python3 scripts/pretrain.py \
  --method simsiam_dinov2 --backbone dinov2_vitb14 \
  --initial-weights ~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth \
  --image-root ~/work/tone-mirror-20261103/full/mirror \
  --transformed-image-root ~/work/tone-mirror-20261103/full/tone_s42 \
  --transformed-manifest-path ~/work/tone-mirror-20261103/full/tone_s42/tone_transform_manifest.json \
  --transformed-data-revision d28901157e7f87bd6ed237071dadda0039240a064e59e9e742719f3883db4852 \
  --data-revision f4fcee05cd4e4f884d31bb6a9bc08fc7225817d2529c8fa22eb1b2ab261462f4 \
  --tone-consistency-weight 0.194 \
  --split-hash 8a83773bcbad5d9ba82e2aae103bede195cab7881975d9db6d03d09305897863 \
  --max-steps 50 --batch-size-per-gpu 16 --seed 42 --precision bf16 --num-workers 0 \
  --output-dir /tmp/tone-probe-verify
python3 -c "import json; d=json.load(open('/tmp/tone-probe-verify/run.json'))['execution']; print(d['ssl_loss'], d['total_loss'])"
```
