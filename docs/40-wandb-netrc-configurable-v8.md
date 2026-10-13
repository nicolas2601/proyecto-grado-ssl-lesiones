# 40. Credenciales W&B desde una ruta netrc configurable

## Cambio y alcance

La verificación del clúster encontró una credencial válida en `.secrets/netrc`,
pero el logger solo comprobaba `~/.netrc`. Se añade `WANDB_NETRC_FILE` para
seleccionar un archivo privado sin moverlo, copiarlo ni ejecutar `wandb login`.
El cambio se aplicó únicamente en la copia aislada `repo_w3_v8_20261013`,
creada desde `experiment/modular-audit-v8` en `8da61a0`.

Prioridad: `WANDB_API_KEY`, modo offline, ruta explícita `WANDB_NETRC_FILE`,
y finalmente `~/.netrc` cuando no hay ruta explícita. Una ruta explícita
inválida no utiliza silenciosamente otro archivo. En POSIX se rechazan archivos
con permisos de grupo u otros usuarios; `600` y `400` son válidos.

## Uso en el clúster

Desde la copia aislada, configurar la ruta sin incluir el contenido de la clave:

```bash
export WANDB_NETRC_FILE="$PWD/../.secrets/netrc"
```

Usar el Python del entorno existente `../.venv/bin/python` y el proyecto W&B
correspondiente a cada experimento mediante `--wandb-project`. La variable
contiene solo una ruta; no debe incorporarse la clave a scripts, argumentos,
configuraciones de experimentos o commits.

## Manejo de la credencial

El logger lee únicamente la entrada `api.wandb.ai`. La clave se entrega al SDK
mediante el entorno del proceso durante `client.init`; un bloque `finally`
restaura el valor anterior incluso si falla la inicialización. Un bloqueo
serializa las inicializaciones de este logger. La credencial se conserva en
memoria durante la vida del sink para ocultarla en advertencias posteriores;
también se elimina de valores anidados de configuración y registros de métricas.

No se escribe una credencial nueva ni se cambia el archivo netrc. La variable
temporal pertenece al proceso Python y a los procesos que el SDK inicie durante
su inicialización; no modifica la configuración persistente de la terminal.
Como con cualquier credencial en memoria, el proceso y sus hijos autorizados
deben permanecer en el entorno de confianza del proyecto.

## Validación realizada

Fecha local: 2026-10-12, America/Bogota.

- `tests/test_seg_wandb_logging.py`: **42 pruebas pasaron**, en 2,15 segundos,
  con la `.venv` existente. Incluye escritura offline con el SDK real y pruebas
  de ruta configurable, permisos, archivos inválidos, prioridad, restauración
  del entorno y ocultación de la clave al fallar init/log/finish.
- Dos advertencias del entorno de pruebas: opciones `timeout` y `timeout_method`
  no reconocidas. No hubo fallos; no se instalaron ni modificaron dependencias.
- Credencial real: detección y autenticación de lectura con W&B **correctas**,
  cliente `0.30.0`; la consulta no creó una corrida.
- Después de la consulta: entorno restaurado, hash del archivo de credenciales
  sin cambios y estado de `~/.netrc` sin cambios.
- No se lanzaron entrenamientos, no se enviaron imágenes ni se modificaron
  las copias anteriores del clúster.

La consulta real valida la autenticación a través del mismo contexto de
credenciales utilizado por el logger. La creación online de una corrida y la
persistencia de sus métricas se comprobarán al iniciar el piloto autorizado;
esta verificación no representa resultados de W3.

Evidencia local fuera del repositorio: `audit_v8_sync_20261013/logger_tests.log`
y `logger_real_credential_check.json`. No contienen claves.
