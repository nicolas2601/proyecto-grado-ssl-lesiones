# 06 — Problemáticas abiertas y resueltas

Registro honesto de los problemas que hemos encontrado. Es el corazón de la defensa: mostrar que
entendemos los límites.

## 🟢 CERRADA — Contradicción del linear probe (resuelta 2026-07-07)

El run anterior mostraba un patrón raro: el linear probe SSL daba **0.264** (colapsado) vs 0.506 de
ImageNet. Ya está **resuelto y explicado**:

- **Causa confirmada:** *BN drift*. El SSL corría a 128 px y la evaluación/probe a 224 px → las
  estadísticas de BatchNorm no coincidían → colapso **aparente** (no real).
- **Fix aplicado y confirmado:** `SSL_RES = 224` (igual que fine-tune y eval). El linear probe SSL
  subió de **0.264 → 0.4509** — número sano y coherente. **El colapso desapareció.**
- **Veredicto final (honesto):** con 224 px la contradicción se disuelve, pero el SSL **no supera** a
  ImageNet en macro-F1 (queda por debajo en el probe, empate en fine-tune). **SÍ gana en AUC**
  (fine-tune +0.0087; probe 100 % +0.0132). Conclusión: no era un bug, es el **límite real del SSL
  con cómputo de estudiante**. Se reporta así — el aporte se reposiciona hacia protocolo + fiabilidad
  + hallazgo AUC, no hacia "SSL gana". Ver [11 — Resultados finales](11-resultados-finales.md) §9.

## 🟢 RESUELTA — Fallas graves del notebook v1

El v1 daba macro-F1 0.873, pero era **mentira**. Tres fallas:
1. **SSL decorativo:** entrenaba un backbone SimCLR (ResNet18) y lo guardaba, pero el clasificador
   arrancaba de `resnet50(ImageNet)` y **nunca lo cargaba**. El 0.873 era ResNet50 supervisado puro.
2. **Data leakage:** split por `image_id` → misma lesión en train y val.
3. **KPIs no medidos:** sensibilidad de melanoma por debajo de 0.80; ECE y AUC ni calculados.

Con split honesto (por `lesion_id`) el número real baja a ~0.74. **El 0.873 estaba inflado.**

## 🟢 RESUELTA (conceptual) — SSL desde cero pierde

Práctica 1: SimCLR entrenado desde cero (cómputo de estudiante) dio F1 0.285, **peor que un
backbone random** (0.318). No es un bug — es lo esperado. SSL desde cero necesita millones de
imágenes + batches enormes. **Conclusión:** nunca desde cero; siempre continuar un backbone
pre-entrenado (domain-adaptive).

## 🟢 VERIFICADA OK — Shortcut learning (artefactos)

HAM10000 tiene pelos, reglas y marcadores; el modelo podría "hacer trampa" mirando el artefacto.
**Control aplicado:** Grad-CAM sobre melanomas. **Resultado (2026-07-07):** el calor cae **SOBRE la
lesión** (p. ej. ISIC_0026531 p=0.84, ISIC_0032396 p=0.54), **no** en pelos ni reglas. En el caso de
baja confianza (p=0.20) el modelo duda, que es honesto. **Sin evidencia de shortcut learning** → es
evidencia positiva de fiabilidad. Ver [07](07-fiabilidad-y-sesgo.md) y [11](11-resultados-finales.md) §6.5.

## 🟡 VIGILAR — Sesgo por fototipo (fairness)

HAM es piel clara (Fitzpatrick I–III). No podemos afirmar rendimiento en piel oscura sin datos.
Dato duro de la literatura: un modelo (DeepDerm) tuvo sensibilidad 0.69 en piel clara vs **0.23 en
piel oscura**. Plan: test set **BOSQUE** (165 imgs de Bogotá con fototipo, arXiv 2507.14176) como
trabajo futuro. Ver [07](07-fiabilidad-y-sesgo.md).

## ⚙️ OPERATIVAS (Kaggle / entorno)

- **MCP de Kaggle no autoriza** en Claude Code (cliente incompatible). Se corre a mano.
- **`math.so` corrupto** en `~/Descargas`: rompe `import math` de Python si el cwd es esa carpeta.
  Correr scripts desde otro directorio.
- **Cómputo limitado:** T4 de estudiante. El entrenamiento definitivo depende de las GPUs de la UNAB.
- **Sesión de Kaggle:** para runs largos usar **Save & Run All (Commit)**, no draft (evita que se
  muera al cerrar el navegador).

## 🔐 SEGURIDAD — pendiente

- Rotar el **token de Kaggle (KGAT)** que quedó expuesto en una corrida previa.
