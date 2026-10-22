# Fuentes y ramas

## Decisión de consolidación

La rama `proyecto-de-grado` se creó desde `proposed-cluster` en el commit
`39e0b19`. Esa rama ya contenía el árbol de v6 y añadía la evidencia más
reciente del pipeline propuesto: `resultados/`, los documentos 22–24, los
artefactos de segmentación y los tests de trazabilidad.

No se hizo un merge bruto de las ramas divergentes. Las ramas originales se
mantienen intactas para poder comparar, recuperar o auditar cualquier cambio.

## Matriz de ramas

| Rama | Punta verificada | Papel | Estado |
|---|---|---|---|
| `proyecto-de-grado` | `bcbe9b4` + consolidación documental | Fuente de trabajo unificada | Activa |
| `proposed-cluster` | `39e0b19` | Resultados finales del pipeline propuesto | Fuente de evidencia de origen |
| `feature/v6-gpu-ssl-pretraining` | `33b64fd` | Código v6, DDP, preentrenamiento y segmentación integrada | Línea técnica conservada |
| `feature/proposed-pipeline-tone` | `c30b063` | Implementación experimental de adaptación por tono | Línea de investigación conservada |
| `feature/fitzpatrick-domain-adaptation-informe` | `73224ca` | Informe y estratificación por tono | Evidencia conservada |
| `feature/v5-ssl-only-evaluation` | `1da0331` | Bancos congelados v5 | Histórico |
| `main` | `202ea73` | Línea v4 de julio | Histórico, no fuente final |

Las puntas locales y sus remotos configurados estaban alineados durante la
auditoría. También existen referencias remotas antiguas de PR1–PR4 de
Fitzpatrick; no se podaron ni se borraron.

## Qué leer de cada línea

- Para el resultado científico más reciente: [`../resultados/README.md`](../resultados/README.md)
  y [`24-analisis-final-resultados-y-escenarios-ada.md`](24-analisis-final-resultados-y-escenarios-ada.md).
- Para el código del preentrenamiento propio: [`../src/dermssl/pretrain/`](../src/dermssl/pretrain/)
  y [`16-plan-v6-pretraining-gpu.md`](16-plan-v6-pretraining-gpu.md).
- Para los bancos históricos: [`../results/README.md`](../results/README.md) y
  [`19-resultados-bancos-congelados.md`](19-resultados-bancos-congelados.md).
- Para la segmentación: [`../plan-segmentacion/camino-b-cnn/resultados/README.md`](../plan-segmentacion/camino-b-cnn/resultados/README.md).
- Para la app de demostración: [nicolas2601/lupa](https://github.com/nicolas2601/lupa); el servicio de
  inferencia V9 está en la rama [`phone-web`](https://github.com/nicolas2601/proyecto-grado-ssl-lesiones/tree/phone-web/demo/backend).

## Worktrees auditados

| Ruta | Rama | Nota |
|---|---|---|
| `Documentos/proyecto-grado-ssl-lesiones` | `feature/v6-gpu-ssl-pretraining` | Checkout técnico original, limpio durante la auditoría. |
| `Documentos/proyecto-grado-segwork` | `feature/fitzpatrick-domain-adaptation-informe` | Checkout de informe; conservaba un `lefthook.yml` sin seguimiento. |
| `Documentos/proyecto-grado-v6work` | `proposed-cluster` | Contenía el demo y sus tests sin seguimiento; fueron incorporados a esta rama. |
| `Documentos/proyecto-grado-consolidado` | `proyecto-de-grado` | Worktree aislado para la consolidación. |

## Política de mantenimiento

1. Crear cambios nuevos sobre `proyecto-de-grado` o sobre una rama corta que
   nazca de ella.
2. Mantener `resultados/` como carpeta de evidencia final y `results/` como
   compatibilidad/histórico de los runners existentes.
3. No borrar ramas antiguas ni ejecutar `remote prune` sin revisar primero sus
   commits únicos.
4. Si dos ramas discrepan, conservar ambos artefactos y registrar la decisión
   en este archivo o en [`04-decisiones-tecnicas.md`](04-decisiones-tecnicas.md).
5. Antes de publicar una cifra, comprobar que el enlace, el JSON fuente y el
   commit estén presentes en la rama consolidada.

## Comandos de inspección seguros

```bash
git branch -vv
git worktree list
git log --oneline --decorate --all -20
git show proposed-cluster:resultados/README.md
git diff proyecto-de-grado..feature/v6-gpu-ssl-pretraining --stat
```

Estos comandos solo inspeccionan. La limpieza de ramas requiere una decisión
separada y no forma parte de esta consolidación.
