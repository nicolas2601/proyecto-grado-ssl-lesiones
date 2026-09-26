# Proyecto de Grado UNAB — Clasificación de Lesiones Cutáneas con Aprendizaje Autosupervisado (SSL)

Repositorio privado de trabajo para investigar la clasificación experimental de lesiones cutáneas mediante aprendizaje autosupervisado. No es un dispositivo de diagnóstico.

## Objetivos específicos

1. Caracterizar conjuntos de datos de imágenes dermatológicas representativos de lesiones cutáneas relevantes para el contexto clínico del departamento de Santander. — Estado: Completado.
   Ver [datasets](docs/02-datasets.md) y [contexto](docs/01-contexto-y-objetivos.md).
2. Diseñar un algoritmo de clasificación de lesiones cutáneas basado en aprendizaje autosupervisado utilizando imágenes dermatológicas. — Estado: Completado.
   Ver [pipeline propuesto](docs/22-pipeline-propuesto-segmentacion-tono-clasificador.md) y [plan v6](docs/16-plan-v6-pretraining-gpu.md).
3. Evaluar el desempeño del algoritmo propuesto mediante métricas reportadas en la literatura científica para tareas de análisis de imágenes médicas. — Estado: En curso (versión v7, ver docs/27).
   Ver [resultados finales](resultados/README.md) y [análisis final](docs/24-analisis-final-resultados-y-escenarios-ada.md).

## Arquitectura

![Arquitectura completa del sistema](assets/figuras/fig11_infografia_arquitectura.png)

Flujo de datos, segmentación, estimación de tono, adaptación SSL, clasificación y evaluación. Ver la [arquitectura detallada](docs/GUIA.md#arquitectura-del-sistema).

## Equipo

| Nombre | Rol |
|---|---|
| Nicolás Santiago Moreno Monroy | Autor |
| María Paula Saavedra | Autora |
| Andrés Jerez | Director |
| Karen Sánchez — KAUST | Asesora |
| UNAB — Ingeniería de Sistemas | Institución |

## Dónde está todo

| Recurso | Enlace |
|---|---|
| Documentación | [docs/README.md](docs/README.md) |
| Guía del repositorio | [docs/GUIA.md](docs/GUIA.md) |
| Resultados | [resultados/README.md](resultados/README.md) |
| Baselines | [resultados/baselines/README.md](resultados/baselines/README.md) |
| Demo | [demo/README.md](demo/README.md) |
| Cluster | [cluster/README.md](cluster/README.md) |
| Assets | [assets/README.md](assets/README.md) |
| Plan v7 | [docs/27](docs/27-plan-v7-mejoras-ssl.md) |
