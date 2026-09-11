# 20. Falsear los guards: por que un test puede no existir aunque pase

Este documento sale de un dia, el 2026-09-11, en el que las dos sesiones que
trabajaban el banco de fine-tuning encontraron **siete** tests propios que
pasaban por la razon equivocada. Los siete los habiamos escrito nosotras, casi
siempre el mismo dia que arreglabamos el bug que ese test supuestamente
protegia, que es exactamente cuando uno esta mas convencido de que el test es
bueno.

No es un documento sobre testing en general. Es sobre una clase concreta de
test: el **guard estatico**, el que lee un archivo y verifica que algo este
ahi. Son baratos, se escriben en dos minutos, y por eso se escriben mal.

## La regla

Un guard no verifica un mecanismo hasta que se lo falsea. Y hacen falta **dos**
falseos, no uno:

| Falseo | Que se hace | Que pregunta responde |
|---|---|---|
| 1 | Romper o borrar el mecanismo | ¿Existe el guard? |
| 2 | Reemplazarlo por algo plausible pero incorrecto | ¿Es este el guard fuerte, o hay uno mejor en otro lado? |

El segundo es el que casi nadie hace, y es el que encuentra los duplicados
debiles: un duplicado cae igual que el original en el escenario obvio, asi que
el falseo 1 no lo distingue.

Cuesta dos minutos. Si el test no cae, el test no existe: solo lo parece.

## Las cuatro formas de guard falso

### 1. Busca un token que el propio mecanismo declara

```python
assert "FEATURE_DIMS" in code        # el diccionario que lo declara ya satisface esto
```

El caso real: ese guard protegia el bug de extraccion de VGG19, el unico error
del proyecto que ya costo re-correr dos bancos enteros. El `assert DIM ==
FEATURE_DIMS[MODEL]` vive en 48 notebooks; **se borro de uno solo**, el de
VGG19, y los siete tests de `test_feature_extraction_point.py` pasaron igual.
El mecanismo con mas historial de fallar era el peor protegido, y bastaba
romperlo en un archivo para que la suite siguiera verde.

> Este parrafo decia antes "se borro de los 16 notebooks". Era falso dos veces:
> el falseo se hizo en uno, y el assert vive en 48, no en 16 — ese numero es el
> de los jobs del banco de fine-tuning, otra cosa. El error entro por transcribir
> el reporte de quien hizo el experimento en vez de verificar su alcance, que es
> justo lo que este documento pide no hacer. Se corrigio al revisarlo entre las
> dos sesiones. La version correcta ademas es **mas** fuerte: que alcance con un
> archivo es peor que si hubieran hecho falta los 48.

```python
assert "assert DIM == FEATURE_DIMS[MODEL]" in code   # ahora exige la comparacion
```

### 2. Busca sobre un blob en vez de sobre la seccion exacta

Dos casos reales el mismo dia:

- El caveat de fuga de ISIC 2019 se buscaba en **todo** el notebook. Al sacarlo
  del `SPLIT_DESC` y dejarlo en un comentario, el test paso. Y `SPLIT_DESC` es
  el unico lugar donde importa, porque es el string que viaja **dentro** del
  results JSON hasta quien lee los numeros. Un comentario no viaja a ningun lado.
- La fila de resultados del README se verificaba con `numero in readme`. Un
  README con veinte tablas contiene casi cualquier valor de tres decimales en
  algun lado: se falseo la fila de alexnet a 0.999 y el test paso, porque
  encontraba el valor original en la tabla de recall por clase de otro banco.

El arreglo es el mismo en los dos: extraer la seccion exacta y comparar ahi.

### 3. Matchea su propio comentario explicativo

```python
assert "CI_REQUIRE_FULL_DEPS" in workflow   # el comentario que lo explica basta
```

Borrar el bloque `env:` no hacia fallar nada. El arreglo: exigir el literal
completo, `CI_REQUIRE_FULL_DEPS: "1"`.

### 4. Duplica uno existente, mas debil

Dos casos, uno de cada sesion, con la misma forma:

- Un guard del arreglo SLURM que verificaba el rango de jobs. Ya existia otro
  que verificaba el rango **y** el `%4`, que es el limite de cuatro GPU del
  laboratorio.
- Un guard del manifiesto que caia con el bug original pero no cuando se
  agregaba un tercer dataset. El guard que ya existia caia en los dos casos.

En los dos, lo peor no es la redundancia: es **donde vivian**. Un duplicado
debil en otro archivo se lee como cobertura, y quien abra ese archivo va a
creer que el contrato esta protegido ahi.

Los dos se borraron, dejando un comentario que apunta al guard real.

## Un guard nuevo tambien puede abrir un hueco

Al tapar un agujero con un guard mejor, hay que correr los dos falseos sobre
el guard nuevo. En este mismo dia paso:

El guard de la URL de clonado solo rechazaba el placeholder `<org>`. El falseo
2 mostro que un usuario con typo (`nicolas26` por `nicolas2601`) pasaba y le
entregaba un 404 a quien clonara. Se reescribio para comparar contra el remoto
real de git... y el nuevo capturaba el typo pero **ya no** el `<org>`, porque
el regex no incluia `<>` y el placeholder ni siquiera se encontraba.

Se habia cambiado un hueco por otro. Solo se vio corriendo los dos falseos
contra el reemplazo.

## Como se hace en la practica

```python
import subprocess
from pathlib import Path

p = Path("<archivo con el mecanismo>")
orig = p.read_text()
for reemplazo, etiqueta in CASOS:
    try:
        p.write_text(orig.replace(MECANISMO, reemplazo, 1))
        r = subprocess.run(["python3", "-m", "pytest", "-q"], capture_output=True)
        print(etiqueta, "->", "FALLA (cubierto)" if r.returncode else "PASA (HUECO)")
    finally:
        p.write_text(orig)          # el finally no es opcional
```

El `finally` importa: un falseo que no se revierte deja el repositorio roto de
una forma dificil de notar, porque los tests vuelven a pasar en cuanto se
revierte a mano el archivo equivocado.

## Relacion con el resto del proyecto

Complementa la regla de que los datos publicados se generan y no se transcriben
(ver el generador `scripts/build_results_tables.py` y los tests de frescura).
Esa regla dice: *si se publica, un test compara*. Esta agrega: **y al test hay
que falsearlo, porque comparar mal es peor que no comparar** — un guard falso
no solo no protege, ademas convence de que el area esta cubierta y desvia la
atencion a otro lado.
