# 29. Plan v7.1: sacar más de las finales existentes

**Registrado el 2026-09-28, antes de correr cualquier cosa.** Parte de la revisión de literatura; los papers citados están verificados.

## 1. Regla central

Toda selección (α, protocolo del probe, resolución) se hace en el **holdout interno**: el 20 % del train de C1, agrupado por `lesion_id` y excluido del pool de pretraining (docs/28 §4.1). La validación se toca **una sola vez al final**, con la configuración elegida.

Límites que se declaran desde ya:
- El holdout ya se usó para elegir A3 y B1 en los pilotos, así que elegir de nuevo sobre él suma optimismo de selección en el holdout. La lectura en val no lo tiene.
- Val ya se leyó con las finales de la semilla 42 y de la 43, y se va a leer con C1-v7 (docs/28 §5.2). La lectura v7.1 es una más, sobre modelos derivados de los mismos; se reporta como tal y el resultado primario de docs/28 §5.1 no cambia.

## 2. Bloques, en el orden de GPU

Antes van S-diag (docs/27), C1-v7 (docs/28 §5.2) y S4-UNet (docs/27). El costo de cada bloque se envía antes de lanzarlo.

### 2.1 WiSE-FT y seed soup (Wortsman et al., arXiv:2109.01903 y arXiv:2203.05482)

Todo se hace sobre las finales existentes, sin entrenar.
- θ = (1 − α)·público + α·adaptado, con α ∈ {0,2; 0,4; 0,6; 0,8; 1,0}, para A\* y B\*.
- **Soup:** promedio de los pesos de las semillas 42 y 43 del mismo brazo. También se evalúa el soup combinado con WiSE (mismos α).
- **Test previo, obligatorio:** en el holdout, α = 0 reproduce las métricas del público y α = 1 las de la final, con tolerancia 1e-6. Si no, no se sigue.

### 2.2 Protocolo del probe (DINOv2, Oquab et al., arXiv:2304.07193)

- **Features:** CLS | CLS + media de los parches | concatenación del CLS de las 4 últimas capas.
- **Pesos de clase:** ninguno | balanced.
- Todo en el holdout. La ganadora se reporta solo como SECUNDARIA en val. El protocolo primario (CLS, `logreg`, C = 1) no cambia.

### 2.3 Resolución

Extracción a 224 contra 448 con los mismos pesos, en el holdout. Es un diagnóstico secundario.

### 2.3b Métricas clínicas secundarias (registradas el 2026-09-28, antes de correr)

Para cada configuración evaluada en el holdout (α, soup, protocolo, pesos de clase, resolución) se reportan también la sensibilidad de melanoma (recall de `mel`) y el AUC de melanoma contra el resto, ambas con IC bootstrap. Con `class_weight = balanced` se miran en particular: es la palanca más directa para bajar la confusión de melanoma con nevus, que en la v7 es el error dominante (sensibilidad de melanoma entre ~0,43 y 0,50 en val). Son secundarias y no cambian la regla de selección, que sigue siendo el macro-F1 de §2.4.

### 2.4 Regla de combinación

En el holdout se elige la mejor configuración de cada brazo por macro-F1; si |Δ| < 0,005, gana la de menor ECE (igual que docs/28 §4.1). Después se corre una sola evaluación en val de esa configuración contra el público, con bootstrap pareado. Se reportan siempre las dos cosas: el primario v7 intacto y la lectura v7.1.

### 2.5 Semilla 44

A\* y B\* con la semilla 44, una GPU cada una, con la misma receta, la revisión f1c2053 y el exclude 95d21013. Es una réplica adicional y se evalúa con la regla de docs/28 §4.1 extendida a tres semillas: se reporta cada semilla, la media y el rango, sin elegir ninguna. La varianza entre semillas (~0,04 en macro-F1 de A\*) es del tamaño del efecto; más semillas le dan peso a la conclusión y alimentan el soup.

### 2.6 Prueba externa sin entrenar: PAD-UFES-20 (Pacheco et al., arXiv:2007.00478; CC BY 4.0; Mendeley zr7vgbcyr2)

- Se baja al pod y se verifica el conteo (2.298 imágenes, 1.373 pacientes) y la columna de paciente antes de usarlo.
- El probe se entrena en el train de HAM (como en C1) y se evalúa en PAD solo en la intersección registrada:
  - **primaria:** bcc, mel y nv (NEV en PAD);
  - **secundaria:** además, akiec contra ACK + SCC.
- Público contra A\* contra B\*, con bootstrap agrupado por paciente.
- Es robustez fuera de dominio (PAD son fotos clínicas de smartphone, no dermatoscopia) y va a la tesis tal cual, sin ajustar nada.

## 3. Fase 2 (fuera por ahora, anotada)

- ExPLoRA / LoRA de pretraining extendido (arXiv:2406.10973).
- Sumar PAD al pool SSL.
- Fine-tune supervisado en PAD: solo demo, fuera de la tesis.

## 4. Trazabilidad

Todo va a W&B, a Engram y a las tablas generadas (`resultados/clasificacion/TABLA_CLASIFICADOR.md`). Se hace push al cerrar cada bloque.
