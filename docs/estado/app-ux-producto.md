# Sesion: app-ux-producto  |  2026-10-30

## Tarea

Investigar y proponer los cambios pertinentes de la app desplegada en https://skin.tikno.pro (repo `skin-model-pwa`, nombre en produccion «Image study — research prototype») en producto y
experiencia de usuario, sin tocar codigo.

## Contexto minimo (no leer mas que esto)

La app es un PWA de apoyo al tamizaje de lesiones cutaneas. No es un dispositivo de diagnostico.
Consume una API de inferencia externa que devuelve probabilidades sobre siete clases de
HAM10000. El usuario objetivo vive en Santander, Colombia, con acceso limitado a dermatologia.

Dato duro que condiciona todo: el modelo que hoy sirve la API **pierde** contra un baseline
publico cuando la foto es de celular (macro-F1 -0,107 y -0,131 con intervalos de confianza que
excluyen el cero). La app debe comunicar incertidumbre, no certeza.

## Entra

- `C:/Users/mpaul/Documents/skin-model-pwa/PRODUCT.md` (y la app en vivo: https://skin.tikno.pro)
- `C:/Users/mpaul/Documents/skin-model-pwa/DESIGN.md`
- `C:/Users/mpaul/Documents/skin-model-pwa/src/` (solo lectura, para entender el flujo actual)
- `https://derm.nicolasmoreno.site/docs` (contrato de la API, solo lectura)

No leer el resto del repositorio `ProyectoDeGrado`.

## Sale

Un unico archivo: `docs/estado/app-ux-producto.reporte.md`, con esta estructura:

1. **Flujo actual**, reconstruido paso a paso desde el codigo, con rutas de archivo.
2. **Hallazgos**, cada uno con: que se observo, donde, por que importa clinicamente.
3. **Propuestas**, cada una con: cambio concreto, justificacion, y costo estimado en
   esfuerzo (bajo / medio / alto).
4. **Riesgos de comunicacion clinica**: donde la interfaz puede inducir a un usuario a creer que
   recibio un diagnostico.

## Criterios (binarios)

1. Cada hallazgo cita un archivo y una linea, o una respuesta real de la API.
2. Ninguna propuesta altera el modelo, su preprocesamiento ni las cifras reportadas.
3. Se evalua explicitamente como la interfaz comunica incertidumbre y limitaciones.
4. Se evalua accesibilidad: contraste, tamano de area tactil, lectura con lector de pantalla,
   uso con una sola mano.
5. Se evalua el caso sin conexion y el caso de foto de mala calidad.
6. El reporte distingue lo verificado de lo no verificado, con el comando o la ruta que lo
   demuestra.

## Fuera de alcance

- No escribir ni modificar codigo.
- No tocar el repositorio `ProyectoDeGrado`.
- No opinar sobre el modelo, la arquitectura de red, las metricas ni el pipeline de ML.
- No proponer marca, logotipo ni identidad visual nueva.
- No desplegar nada ni llamar a `POST /api/predict` con imagenes de personas.

---
## Reporte  (lo completa la sesion al terminar)
Verificado: <con que comando>
No verificado: <y por que>
Bloqueos:
