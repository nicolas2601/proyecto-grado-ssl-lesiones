# 06 — Problemáticas abiertas y resueltas

Registro honesto de los problemas que hemos encontrado. Es el corazón de la defensa: mostrar que
entendemos los límites.

## 🔴 ABIERTA — Contradicción del linear probe

El último run mostró un patrón raro **sin resolver**:

- Full fine-tune SSL "ganó" por +0.035 (macro-F1 0.771) **PERO** el linear probe mostró el SSL
  **peor** (F1 0.264 vs 0.506 de ImageNet).
- El +0.035 del fine-tune es probablemente **ruido** (osciló −0.012 → +0.035 entre corridas).
- **Causa probable:** SSL entrenado a 128px mientras la evaluación/probe era a 224px → **BN drift**
  (BatchNorm con estadísticas de otra resolución) y posible colapso aparente.
- **Fix aplicado (sin confirmar aún):** `SSL_RES = 224` (igual que el resto).
- **Qué falta:** correr y ver si a 224px el linear probe del SSL recupera (dF1 ≥ 0).
  - Si recupera → contribución del SSL **rescatada** (la figura clave de la tesis).
  - Si no → reportar como **límite honesto** y quedarse con el mejor modelo + AUC + fiabilidad.

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

## 🟡 VIGILAR — Shortcut learning (artefactos)

HAM10000 tiene pelos, reglas y marcadores. El modelo puede "hacer trampa" mirando el artefacto en
vez de la lesión. **Control:** Grad-CAM sobre melanomas — el calor debe caer SOBRE la lesión.
Si mira artefactos → reportar como riesgo de fiabilidad.

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
