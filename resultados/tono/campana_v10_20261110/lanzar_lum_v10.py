"""Campana `V10_C1M_LUM`: el operador tonal corregido al eje de LUMINANCIA.

QUE CAMBIA RESPECTO DE `V10_C1M`, Y ES UN SOLO FACTOR
-----------------------------------------------------
Solo la DIRECCION del desplazamiento en Lab. La dosis sorteada es identica
item por item (verificado, `max|dif| = 0,0` sobre 400 filas), la elegibilidad
la sigue decidiendo el modulo congelado, la mascara del teacher 026 y el
acople `w(x) = alpha + (1-alpha)*m` son los mismos.

    V10_C1M      dL* = -0,24*db*,  da* = 0,  db* = 0,50*ita_delta
    V10_C1M_LUM  dL* = -0,40*ita_delta,  da* = 0,  db* = 0

POR QUE, CON LA CIFRA QUE LO MOTIVA
-----------------------------------
`causa-fisica-ddi-20261109.json`, 656 casos de DDI. A dosis nominal 25 grados,
la geometria `b*` entrega 15,0 grados de ITA en FST I-II y **3,6 en FST V-VI**.
La causa es analitica: `d(ITA)/d(b*) = -(L*-50)/((L*-50)^2 + b*^2)` se anula
cuando `L* -> 50`, y la banda FST V-VI de DDI tiene ITA natural -4,02 grados.
El tratamiento no llegaba al grupo que debia beneficiar. Con la geometria
L*-dominante la entrega pasa a 25,07 / 21,87 / 20,96: la dispersion entre
bandas cae de **4,17x a 1,20x** (`verificacion-dosis-luminancia-20261109.json`).

YA VERIFICADO EN CPU ANTES DE ESTA CAMPANA
------------------------------------------
`smoke-lum-cpu-20261109.json`, 158 vistas reales con mascaras del 026:
L0 dosis cero no cambia un bit; L1 el eje es luminancia (`A_b` max 0,108 y
razon `A_L` 0,9995); L2 entrega 20,84 grados de ITA mediana; L3 la mascara
entra (`m=0` congela con p95 = 0,0 exacto, `m=1` iguala con razon 0,967);
L4 dosis pareada con C1M (`max|dif| = 0,0`); L5 el piso `L* >= 28` se respeta
en el 100 %; L6 el cambio llega a los pixeles (25 niveles de 8 bits minimo).

L7 se REFUTO y se reporta: predije que el piso costaria MAS atenuacion que en
C1M y salio IGUAL (10,1266 % en los dos). El piso no cambia CUANTAS vistas se
atenuan, cambia CUANTO: recorta mas alla del gamut en el 14,17 % de las vistas
y cuesta el **2,4 %** de la dosis total. No es una guarda muerta ni se come la
dosis; la dosis media entregada es 22,96 de 25 (91,8 % del nominal).

CRITERIOS DEL SMOKE PAREADO, DECLARADOS ANTES DE MEDIR
-------------------------------------------------------
W0  EL BRAZO APLICA ALGO. `applied > 0` en `V10_C1M_LUM`. Es el criterio que
    faltaba el 2026-11-08, cuando un brazo nuevo quedo fuera de la lista de
    `construir_vista()` y habria entrenado 6.000 pasos sin tono.
W1  REGRESION, Y ESTA VEZ NO ES VACUA. `V10_C1M` se RE-ENTRENA 40 pasos en un
    directorio NUEVO (`smoke-regresion-V10_C1M-s{seed}`) con el modulo `b*` de
    siempre, y su `backbone_sha256` se compara contra el almacenado en
    `smoke-V10_C1M-s{seed}`. Es la prueba de que agregar el brazo y el modulo
    nuevo no perturbo el camino de codigo de los brazos ya corridos.

    POR QUE SE CAMBIO. En la campana `asym`, Z1 leia el sha de
    `smoke-V10_C1M-s{seed}` y lo comparaba contra el sha guardado en ESE MISMO
    archivo, porque `lanzar_entrenamiento` se saltea una corrida cuyo
    `result.json` ya existe. Comparaba un valor consigo mismo: un criterio que
    no podia fallar, el mismo modo de falla que el S6 original y que el cero de
    colisiones de un detector ausente. Aca se re-entrena de verdad.

    **Si falla, se detiene todo: las corridas previas dejarian de ser
    reproducibles.**
W2  EL CAMBIO LLEGA A LOS PESOS. `sha(LUM) != sha(C1M)`, y vale solo con W0.
W3  ELEGIBILIDAD IDENTICA. `applied` igual por semilla. Depende del hash y de
    la geometria del recorte, NO del eje. Una diferencia aqui significa un
    segundo factor.
W4  ATENUACION ACTIVA. `margin_attenuated_views > 0` en LUM.
W5  EL EJE CAMBIO EN EL RUNTIME, no solo en el smoke de CPU:
      W5a  `D_L_piel(LUM) > 2,0 x D_L_piel(C1M)`
      W5b  `D_b_piel(LUM) < 0,25 x D_b_piel(C1M)`
    PREDICCION: la razon `D_L` rondara 3,3x (0,40 contra 0,12 por grado) y
    `D_b` quedara practicamente en cero. Si `D_b` no cae, el modulo que se
    cargo no es el que se cree y hay que detenerse.
W6  CONTRATO DE LESION, en tres partes como en C1M y asym:
      W6a `lesion_conservation_contract == "modificada_con_acople_m"`
      W6b `changed_8bit_views <= lesion_modified_views`
      W6c `applied - lesion_modified_views <= margin_attenuated_views`

PRESUPUESTO. Precedente medido en `estado-campana-asym.json`: 7,31-7,34 horas
de reloj por semilla con las 3 concurrentes sobre 2 GPUs. Esperar lo mismo,
no menos: las 3 semillas no caben en 2 GPUs sin contencion.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from lanzar_c1m_v10 import (  # noqa: E402
    BASE_DIR,
    SEEDS,
    crear_config,
    escribir_estado,
    esperar_procesos,
    evaluar_job,
    leer_telemetria,
)

ACOPLE_B = "/home/jovyan/work/moco-tono-20261107/ablacion/ingenieria/tono_acople_corregido.py"
ACOPLE_LUM = "/home/jovyan/work/moco-tono-20261107/ablacion/ingenieria/tono_acople_luminancia.py"
ARM_LUM = "V10_C1M_LUM"
ESTADO = "estado-campana-lum.json"
CRIT_RAZON_DL_MIN = 2.0
CRIT_RAZON_DB_MAX = 0.25


def lanzar(arm, seed, max_steps, out_dir, cfg_path, gpu_id, log_path, ckpt_every, acople):
    """Como `lanzar_c1m_v10.lanzar_entrenamiento`, mas `--acople-module`.

    Se duplica a proposito en vez de modificar el original: ese archivo es el
    registro de lo que ya corrio y no se toca.
    """
    res_json = Path(out_dir) / "result.json"
    if res_json.exists():
        return None
    chk = subprocess.run(["pgrep", "-f", str(cfg_path)], capture_output=True, text=True, check=False)
    if [p for p in chk.stdout.splitlines() if p.strip()]:
        print(f"  Adoptando proceso en curso para {arm}-s{seed}...", flush=True)
        return ("ADOPTED", None, f"{arm}-s{seed}", res_json, log_path)

    cfg = crear_config(seed, max_steps, out_dir, [2000, 4000] if max_steps == 6000 else [20])
    Path(cfg_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg_path).write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    cmd = [
        sys.executable, str(AQUI / "moco_v10_tono_runtime.py"),
        "--arm", arm, "--config", str(cfg_path),
        "--checkpoint-every", str(ckpt_every),
        "--acople-module", acople,
    ]
    fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return (proc, fh, f"{arm}-s{seed}", res_json, log_path)


def leer_run(nombre):
    j = json.loads((BASE_DIR / "runs" / nombre / "result.json").read_text(encoding="utf-8"))
    return j["backbone_sha256"], j["dose_telemetry"]


def evaluar_criterios_smoke():
    detalle, todo_ok = {}, True
    for s in SEEDS:
        sha_prev, _ = leer_run(f"smoke-V10_C1M-s{s}")          # historico, intacto
        sha_cm, t_cm = leer_run(f"smoke-regresion-V10_C1M-s{s}")  # recien re-entrenado
        sha_lu, t_lu = leer_telemetria(ARM_LUM, s, smoke=True)

        r_dl = (t_lu["D_L_piel_mean_over_applied"] / t_cm["D_L_piel_mean_over_applied"]
                if t_cm["D_L_piel_mean_over_applied"] else 0.0)
        r_db = (t_lu["D_b_piel_mean_over_applied"] / t_cm["D_b_piel_mean_over_applied"]
                if t_cm["D_b_piel_mean_over_applied"] else float("inf"))

        w0 = t_lu["applied"] > 0
        w1 = sha_cm == sha_prev
        w2 = (sha_lu != sha_cm) and w0
        w3 = t_lu["applied"] == t_cm["applied"]
        w4 = t_lu["margin_attenuated_views"] > 0
        w5a = r_dl > CRIT_RAZON_DL_MIN
        w5b = r_db < CRIT_RAZON_DB_MAX
        w5 = w5a and w5b and w0
        w6a = t_lu["lesion_conservation_contract"] == "modificada_con_acople_m"
        w6b = t_lu["changed_8bit_views"] <= t_lu["lesion_modified_views"]
        w6c = (t_lu["applied"] - t_lu["lesion_modified_views"]) <= t_lu["margin_attenuated_views"]
        w6 = w0 and w6a and w6b and w6c

        ok = all((w0, w1, w2, w3, w4, w5, w6))
        todo_ok = todo_ok and ok
        detalle[str(s)] = {
            "W0_el_brazo_aplica_algo": w0,
            "W1_regresion_C1M_bit_a_bit": w1,
            "W2_cambio_llega_a_los_pesos": w2,
            "W3_elegibilidad_identica": w3,
            "W4_atenuacion_activa": w4,
            "W5_el_eje_cambio": w5, "W5a_D_L_sube": w5a, "W5b_D_b_cae": w5b,
            "W6_contrato_lesion": w6, "W6a": w6a, "W6b": w6b, "W6c": w6c,
            "holds": ok,
            "sha_C1M_almacenado": sha_prev,
            "sha_C1M_recalculado": sha_cm,
            "sha_LUM": sha_lu,
            "applied_C1M": t_cm["applied"], "applied_LUM": t_lu["applied"],
            "atenuadas_C1M": t_cm["margin_attenuated_views"],
            "atenuadas_LUM": t_lu["margin_attenuated_views"],
            "gamut_C1M": t_cm["gamut_clipped_pixels"],
            "gamut_LUM": t_lu["gamut_clipped_pixels"],
            "D_L_piel_C1M": t_cm["D_L_piel_mean_over_applied"],
            "D_L_piel_LUM": t_lu["D_L_piel_mean_over_applied"],
            "D_b_piel_C1M": t_cm["D_b_piel_mean_over_applied"],
            "D_b_piel_LUM": t_lu["D_b_piel_mean_over_applied"],
            "razon_D_L_LUM_sobre_C1M": r_dl,
            "razon_D_b_LUM_sobre_C1M": r_db,
        }
        print(f"  s{s}: W0={w0} W1_reg={w1} applied {t_cm['applied']}/{t_lu['applied']} "
              f"D_L {t_cm['D_L_piel_mean_over_applied']:.4f}->{t_lu['D_L_piel_mean_over_applied']:.4f} "
              f"({r_dl:.2f}x)  D_b {t_cm['D_b_piel_mean_over_applied']:.4f}->"
              f"{t_lu['D_b_piel_mean_over_applied']:.4f} ({r_db:.3f}x)  holds={ok}", flush=True)
    return todo_ok, detalle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()
    t_ini = time.time()
    for sub in ("configs", "runs", "logs", "evaluacion"):
        (BASE_DIR / sub).mkdir(parents=True, exist_ok=True)

    estado = {
        "campaign": "v10-c1m-lum-20261109",
        "proposito": (
            "corregir el operador tonal al eje dominado por luminancia, tras medir que la "
            "geometria b* entrega 3,6 grados de ITA de 25 en FST V-VI contra 15,0 en FST I-II"
        ),
        "factor_unico": "solo la direccion Lab; dosis identica item por item y misma elegibilidad",
        "evidencia_previa": {
            "causa": "causa-fisica-ddi-20261109.json",
            "dosis": "verificacion-dosis-luminancia-20261109.json",
            "smoke_cpu": "smoke-lum-cpu-20261109.json",
        },
        "criterios_declarados_antes_de_medir": {
            "W0": "applied > 0 en el brazo nuevo",
            "W1": "el smoke de C1M reproduce bit a bit su sha almacenado",
            "W2": "sha de LUM distinto del de C1M",
            "W3": "applied identico por semilla",
            "W4": "margin_attenuated_views > 0 en LUM",
            "W5": f"D_L(LUM) > {CRIT_RAZON_DL_MIN}x D_L(C1M) y D_b(LUM) < {CRIT_RAZON_DB_MAX}x D_b(C1M)",
            "W6": "contrato de lesion en tres partes",
        },
        "prediccion_registrada": (
            "razon D_L ~ 3,3x (0,40 contra 0,12 por grado) y D_b practicamente cero"
        ),
        "presupuesto_esperado": "7,31-7,34 h de reloj por semilla, 3 concurrentes sobre 2 GPUs",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "smoke_pareado_40_pasos",
        "smoke": {}, "full_runs": {},
    }
    escribir_estado(estado, ESTADO)

    print("[LUM] Fase 1: smoke pareado 40 pasos, regresion de V10_C1M y V10_C1M_LUM "
          "en s42/s43/s44...", flush=True)
    # `V10_C1M` va a un directorio NUEVO para que W1 compare dos corridas y no
    # un archivo contra si mismo. Ver la nota de W1 en el encabezado.
    trabajos = [(arm, s, etq) for s in SEEDS
                for arm, etq in (("V10_C1M", "regresion-V10_C1M"), (ARM_LUM, ARM_LUM))]
    procs = []
    for idx, (arm, s, etq) in enumerate(trabajos):
        acople = ACOPLE_LUM if arm == ARM_LUM else ACOPLE_B
        p = lanzar(arm, s, 40,
                   BASE_DIR / "runs" / f"smoke-{etq}-s{s}",
                   BASE_DIR / "configs" / f"smoke-{etq}-s{s}.json",
                   idx % 2,
                   BASE_DIR / "logs" / f"smoke-{etq}-s{s}.log", 20, acople)
        if p is not None:
            procs.append(p)
    esperar_procesos(procs)

    print("[LUM] Aplicando criterios W0..W6...", flush=True)
    ok, detalle = evaluar_criterios_smoke()
    estado["smoke"], estado["smoke_holds"] = detalle, ok
    escribir_estado(estado, ESTADO)
    if not ok:
        estado["phase"] = "ABORTADA_POR_CRITERIO_DE_SMOKE"
        escribir_estado(estado, ESTADO)
        raise RuntimeError("El smoke pareado no cumple los criterios declarados. Se detiene.")
    print("[LUM] Smoke PASA los 7 criterios en las 3 semillas.", flush=True)
    if args.smoke_only:
        return

    estado["phase"] = "full_3_runs_2_gpus"
    escribir_estado(estado, ESTADO)
    print("[LUM] Fase 2: V10_C1M_LUM 6.000 pasos en s42/s43/s44...", flush=True)
    items = [(ARM_LUM, 42, 0), (ARM_LUM, 43, 1), (ARM_LUM, 44, 0)]
    procs = []
    for arm, seed, gpu in items:
        p = lanzar(arm, seed, 6000,
                   BASE_DIR / "runs" / f"{arm}-s{seed}",
                   BASE_DIR / "configs" / f"{arm}-s{seed}.json",
                   gpu,
                   BASE_DIR / "logs" / f"{arm}-s{seed}.log", 2000, ACOPLE_LUM)
        if p is not None:
            procs.append(p)
    esperar_procesos(procs)

    eval_procs = []
    for arm, seed, gpu in items:
        res = json.loads((BASE_DIR / "runs" / f"{arm}-s{seed}" / "result.json").read_text(encoding="utf-8"))
        tel = res["dose_telemetry"]
        estado["full_runs"][f"{arm}-s{seed}"] = {
            "backbone_sha256": res["backbone_sha256"],
            "wall_hours": res["wall_hours"],
            "applied": tel["applied"],
            "margin_attenuated_views": tel["margin_attenuated_views"],
            "gamut_clipped_pixels": tel["gamut_clipped_pixels"],
            "D_L_piel_mean": tel["D_L_piel_mean_over_applied"],
            "D_b_piel_mean": tel["D_b_piel_mean_over_applied"],
            "D_L_lesion_mean": tel["D_L_lesion_mean_over_applied"],
        }
        print(f"  Entrenado {arm}-s{seed}: sha={res['backbone_sha256'][:12]}... "
              f"({res['wall_hours']*60:.1f} min)", flush=True)
        ep = evaluar_job(arm, seed, gpu)
        if ep is not None:
            eval_procs.append(ep)
    escribir_estado(estado, ESTADO)
    esperar_procesos(eval_procs)

    estado["phase"] = "DONE_SIN_SINTESIS"
    estado["total_minutes"] = (time.time() - t_ini) / 60.0
    escribir_estado(estado, ESTADO)
    print(f"[LUM] Completa en {estado['total_minutes']:.1f} min. "
          f"Falta extraer bancos externos y correr la escalera y T3.", flush=True)


if __name__ == "__main__":
    main()
