"""PRIMARIO CONFIRMATORIO: el eje corregido contra el eje publicado.

MARCO DE PRUEBA JERARQUICO
==========================
Esta campana acumulo mas de 55 contrastes. Reportarlos todos al mismo nivel
seria una nube exploratoria y diluiria el unico aporte que no salio de probar
cosas: la geometria L*-dominante se dedujo ANALITICAMENTE antes de medir, de
la optica de la piel, no de buscar en los datos.

    d(ITA)/d(b*) = -(L*-50) / ((L*-50)^2 + b*^2)  ->  0  cuando  L* -> 50

La banda FST V-VI de DDI tiene ITA natural -4,02 grados, o sea L* ~ 50. Esa
deduccion quedo registrada con criterios P0-P5 declarados ANTES de medir
(`causa-fisica-ddi-20261109.json`, 2026-11-09) y los criterios W0-W7 del brazo
se declararon ANTES de entrenar (`lanzar_lum_v10.py`).

Por eso se adopta una jerarquia fija: UN primario confirmatorio, UN secundario
con compuerta, y todo lo demas formalmente exploratorio.

HONESTIDAD SOBRE LA PRE-ESPECIFICACION
--------------------------------------
**Los resultados de T3 contra el control YA SE CONOCIAN cuando se escribio
este contraste.** Lo que estaba pre-especificado es la HIPOTESIS y su
MECANISMO, no este estadistico en particular. Esto no es un preregistro en
sentido estricto y el informe tiene que decirlo con estas palabras. Lo que si
es auditable: este archivo, con su regla de decision y su margen, se escribio
y se congelo ANTES de calcular el numero.

POR QUE `LUM vs C1M` Y NO `LUM vs T0`
------------------------------------
Es el par natural y el unico que aisla UNA variable. Los dos brazos comparten:
  - la dosis sorteada, identica item por item (`max|dif| = 0,0` sobre 400 filas)
  - la elegibilidad, delegada al mismo modulo congelado (`applied` identico:
    626/626, 628/628, 623/623)
  - la mascara del teacher 026, el acople espacial y la atenuacion por margen
  - la receta, las semillas y el numero de pasos
Lo unico que cambia es la direccion en CIELAB. Contra `T0` cambiarian dos
cosas a la vez: haber tono y que eje.

LA JERARQUIA
------------
  H1  PRIMARIO CONFIRMATORIO.
      D = brecha(LUM) - brecha(C1M), con
      brecha = AUC(FST I-II) - AUC(FST V-VI) en DDI zero-shot.
      Bootstrap PAREADO: los dos brazos se evaluan sobre las MISMAS imagenes
      remuestreadas, 10.000 repeticiones.

      SUPERIORIDAD (bilateral, alfa 0,05): el IC95 de D excluye cero por el
      lado negativo (D < 0 significa que LUM tiene MENOS brecha).

      NO-INFERIORIDAD (unilateral, alfa 0,025): el limite SUPERIOR del IC95
      queda por debajo del margen declarado.

      MARGEN DECLARADO  delta = +0,0139.
      PROCEDENCIA, explicita: es la MITAD del dano medido de `C1M` contra el
      control (+0,0278, IC[+0,0041, +0,0525], `t3-zeroshot-20261109.json`).
      Se usa la convencion de preservar al menos el 50 % del efecto de
      referencia. **El margen sale de estos mismos datos, no de un estandar
      externo; no existe uno publicado para brechas de AUC por fototipo.**
      Eso se declara como limite, no se esconde.

  H2  SECUNDARIO CON COMPUERTA. Solo se interpreta SI H1 demuestra
      superioridad. AUC en FST V-VI, LUM contra C1M, mismo bootstrap pareado.
      La compuerta controla el error de tipo I de la familia sin correccion
      adicional: si H1 no pasa, H2 no se reporta como inferencia.

  EXPLORATORIO. Todo el resto (BOSQUE estratificado, DiD en PAD, escalera
      interna, los contrastes contra `T0`, los lectores `cls`) se reporta como
      evidencia mecanistica exploratoria, sin afirmacion inferencial.

GATE DE REPRODUCCION
--------------------
  R0  Los AUC por brazo y por banda recalculados aqui deben coincidir con
      `t3-zeroshot-20261109.json` con |dif| < 1e-9. **Si no coinciden, el
      arnes cambio y nada de lo que sigue vale.** Se detiene.

PREDICCION REGISTRADA, antes de correr: espero H1 SUPERIOR. La brecha de C1M
es +0,0585 y la de LUM -0,0029; la diferencia puntual es -0,0614, mas del
doble de los contrastes contra el control que ya salieron significativos. Si
sale NEUTRO, el aporte se reporta como no-inferioridad solamente.

SALIDA: `primario-lum-vs-c1m-20261110.json`
"""
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

BASE = Path("/home/jovyan/work/moco-tono-20261107/v10_tono_20261108")
INT = BASE / "evaluacion"
EXT = BASE / "evaluacion_externa"
DDI_DIR = Path("/home/jovyan/work/ddi-20261105")
T3_JSON = BASE / "t3-zeroshot-20261109.json"
SALIDA = BASE / "primario-lum-vs-c1m-20261110.json"

SEEDS = (42, 43, 44)
MODO = "cls4"
MALIGNAS_PAD = {"MEL", "BCC", "SCC", "ACK"}
SEMILLA = 20261109          # la misma que T3, para que R0 pueda reproducir
RESAMPLES_H = 10000
REFERENCIA = "V10_C1M"
CANDIDATO = "V10_C1M_LUM"
CONTROL = "V10_T0"
DELTA_NI = 0.0139           # margen de no-inferioridad, ver encabezado
TOL_R0 = 1e-9


def puntajes_ddi(brazo, por_id):
    """Reproduce exactamente el camino de `t3_zeroshot.main` para un brazo."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    from escalera_v10 import (GRID_C, cargar_npz, etiquetas_pad,
                              metricas_binarias, normaliza)

    acum = []
    for s in SEEDS:
        Xtr, ids_tr, _ = cargar_npz(INT / f"{brazo}-s{s}" / f"pad_adapt-{MODO}.npz")
        ytr = np.asarray([1 if r["diagnostic"] in MALIGNAS_PAD else 0
                          for r in etiquetas_pad(ids_tr)])
        mu, sd = normaliza(Xtr)
        Ztr = (Xtr - mu) / sd

        mejor, mejor_c = -1.0, GRID_C[0]
        cvi = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMILLA)
        for C in GRID_C:
            sc = []
            for tr2, va2 in cvi.split(Ztr, ytr):
                m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
                m.fit(Ztr[tr2], ytr[tr2])
                sc.append(metricas_binarias(ytr[va2], m.predict_proba(Ztr[va2])[:, 1])["balanced_accuracy"])
            if float(np.mean(sc)) > mejor:
                mejor, mejor_c = float(np.mean(sc)), C
        mdl = LogisticRegression(C=mejor_c, class_weight="balanced", max_iter=5000)
        mdl.fit(Ztr, ytr)

        Xte, ids_te, _ = cargar_npz(EXT / f"{brazo}-s{s}" / f"ddi-{MODO}.npz")
        acum.append({"ids": ids_te, "score": mdl.predict_proba((Xte - mu) / sd)[:, 1]})
        print(f"   {brazo}-s{s} listo (C={mejor_c})", flush=True)

    ids0 = acum[0]["ids"]
    for d in acum:
        assert list(d["ids"]) == list(ids0), "orden de ids distinto entre semillas"
    return ids0, np.mean([d["score"] for d in acum], axis=0)


def main():
    from escalera_v10 import metricas_binarias

    man = [json.loads(l) for l in
           (DDI_DIR / "ddi_holdout.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    por_id = {r["image_id"]: r for r in man}

    brazos = [CONTROL, REFERENCIA, CANDIDATO]
    ids0, scores = None, {}
    for br in brazos:
        ids, sc = puntajes_ddi(br, por_id)
        if ids0 is None:
            ids0 = ids
        assert list(ids) == list(ids0), "orden de ids distinto entre brazos"
        scores[br] = sc

    y0 = np.asarray([1 if por_id[i]["malignant"] == "True" else 0 for i in ids0])
    tono = np.asarray([str(por_id[i]["skin_tone"]) for i in ids0])
    idx12 = np.where(tono == "12")[0]
    idx56 = np.where(tono == "56")[0]

    def auc(sel, sc):
        return metricas_binarias(y0[sel], sc[sel])["auc_roc"]

    def brecha(sc, s12, s56):
        a, b = auc(s12, sc), auc(s56, sc)
        return None if a is None or b is None else a - b

    # ------------------------------------------------------------ gate R0
    t3 = json.loads(T3_JSON.read_text(encoding="utf-8"))
    r0_detalle, r0 = {}, True
    for br in brazos:
        alm = t3["por_brazo"][br]
        aqui = {"global": auc(np.arange(len(y0)), scores[br]),
                "12": auc(idx12, scores[br]), "56": auc(idx56, scores[br]),
                "34": auc(np.where(tono == "34")[0], scores[br])}
        difs = {"global": abs(aqui["global"] - alm["global"]["auc_roc"])}
        for b in ("12", "34", "56"):
            difs[b] = abs(aqui[b] - alm["por_banda"][b]["auc_roc"])
        ok = max(difs.values()) < TOL_R0
        r0 = r0 and ok
        r0_detalle[br] = {"max_abs_dif_vs_t3": max(difs.values()), "reproduce": ok}
        print(f"   R0 {br:16} max|dif| = {max(difs.values()):.3e}  -> {ok}", flush=True)
    if not r0:
        SALIDA.write_text(json.dumps({"R0_reproduccion": r0_detalle,
                                      "estado": "ABORTADO: el arnes no reproduce T3"},
                                     indent=2) + "\n", encoding="utf-8")
        raise SystemExit("R0 falla: el arnes no reproduce T3. Se detiene.")

    # ------------------------------------------------ H1 bootstrap pareado
    punto = {br: brecha(scores[br], idx12, idx56) for br in brazos}
    d_punto = punto[CANDIDATO] - punto[REFERENCIA]

    rng = np.random.default_rng(SEMILLA + 7)
    difs_h1, difs_h2 = [], []
    for _ in range(RESAMPLES_H):
        p12 = rng.choice(idx12, len(idx12), replace=True)
        p56 = rng.choice(idx56, len(idx56), replace=True)
        if len(set(y0[p12])) < 2 or len(set(y0[p56])) < 2:
            continue
        bc = brecha(scores[CANDIDATO], p12, p56)
        br_ = brecha(scores[REFERENCIA], p12, p56)
        if bc is not None and br_ is not None:
            difs_h1.append(bc - br_)
        a = auc(p56, scores[CANDIDATO])
        b = auc(p56, scores[REFERENCIA])
        if a is not None and b is not None:
            difs_h2.append(a - b)

    v1 = np.asarray(difs_h1)
    lo1, hi1 = float(np.percentile(v1, 2.5)), float(np.percentile(v1, 97.5))
    superior = hi1 < 0.0
    no_inferior = hi1 < DELTA_NI
    p_dir = float((v1 >= 0).mean())   # proporcion de remuestreos sin efecto o adverso

    h1 = {
        "contraste": f"brecha({CANDIDATO}) - brecha({REFERENCIA})",
        "brecha_referencia": punto[REFERENCIA],
        "brecha_candidato": punto[CANDIDATO],
        "delta_punto": float(d_punto),
        "ic95": [lo1, hi1],
        "resamples_validos": int(v1.size),
        "margen_no_inferioridad": DELTA_NI,
        "superioridad": bool(superior),
        "no_inferioridad": bool(no_inferior),
        "fraccion_remuestreos_adversos": p_dir,
        "veredicto": ("SUPERIOR: el eje corregido reduce la brecha frente al publicado"
                      if superior else
                      "NO INFERIOR, superioridad no demostrada" if no_inferior else
                      "NI SUPERIOR NI NO-INFERIOR al margen declarado"),
    }

    # --------------------------------------------- H2, solo si H1 es superior
    v2 = np.asarray(difs_h2)
    lo2, hi2 = float(np.percentile(v2, 2.5)), float(np.percentile(v2, 97.5))
    h2 = {
        "contraste": f"AUC FST V-VI: {CANDIDATO} - {REFERENCIA}",
        "auc_referencia": auc(idx56, scores[REFERENCIA]),
        "auc_candidato": auc(idx56, scores[CANDIDATO]),
        "delta_punto": float(auc(idx56, scores[CANDIDATO]) - auc(idx56, scores[REFERENCIA])),
        "ic95": [lo2, hi2],
        "compuerta_H1_superior": bool(superior),
        "interpretable": bool(superior),
        "veredicto": (("MEJORA la piel oscura" if lo2 > 0 else
                       "EMPEORA la piel oscura" if hi2 < 0 else "NEUTRO")
                      if superior else
                      "NO SE INTERPRETA: la compuerta H1 no se abrio"),
    }

    rep = {
        "analisis": "primario-confirmatorio-eje-corregido",
        "marco": "jerarquico: H1 confirmatorio, H2 con compuerta, resto exploratorio",
        "declaracion_de_pre_especificacion": (
            "La hipotesis y su mecanismo se declararon antes de medir "
            "(causa-fisica-ddi-20261109.json, criterios P0-P5) y antes de entrenar "
            "(lanzar_lum_v10.py, criterios W0-W7). ESTE estadistico se escribio DESPUES "
            "de conocer T3 contra el control. No es un preregistro en sentido estricto y "
            "el informe debe decirlo asi."
        ),
        "factor_unico": (
            "dosis identica item por item (max|dif|=0,0 sobre 400 filas), elegibilidad "
            "identica (applied 626/626, 628/628, 623/623), misma mascara, mismo acople, "
            "misma receta y semillas; solo cambia la direccion en CIELAB"
        ),
        "procedencia_del_margen": (
            f"delta = {DELTA_NI} es la mitad del dano medido de {REFERENCIA} contra "
            "el control (+0,0278, IC[+0,0041,+0,0525]). Sale de estos datos, no de un "
            "estandar externo: no hay uno publicado para brechas de AUC por fototipo."
        ),
        "R0_reproduccion": r0_detalle,
        "H1_primario": h1,
        "H2_secundario_con_compuerta": h2,
        "exploratorio_no_inferencial": [
            "BOSQUE balacc I-III y IV-VI (los 3 umbrales)",
            "diff-in-diff en holdout PAD, FST I-IV",
            "escalera interna HAM y PAD",
            "contrastes contra V10_T0",
            "lector secundario cls",
            "invariancia al estimulo b* del modulo viejo",
        ],
    }
    SALIDA.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    print("PRIMARIO CONFIRMATORIO  —  eje corregido contra eje publicado")
    print("=" * 78)
    print(f"\n H1  brecha({CANDIDATO}) - brecha({REFERENCIA})")
    print(f"     {punto[REFERENCIA]:+.4f}  ->  {punto[CANDIDATO]:+.4f}")
    print(f"     D = {d_punto:+.4f}   IC95 [{lo1:+.4f}, {hi1:+.4f}]   n = {v1.size}")
    print(f"     superioridad (IC excluye 0 por abajo) : {superior}")
    print(f"     no-inferioridad (limite sup < {DELTA_NI}) : {no_inferior}")
    print(f"     -> {h1['veredicto']}")
    print(f"\n H2  AUC FST V-VI  (compuerta {'ABIERTA' if superior else 'CERRADA'})")
    print(f"     {h2['auc_referencia']:.4f} -> {h2['auc_candidato']:.4f}  "
          f"D = {h2['delta_punto']:+.4f}  IC95 [{lo2:+.4f}, {hi2:+.4f}]")
    print(f"     -> {h2['veredicto']}")
    print(f"\n escrito: {SALIDA}")


if __name__ == "__main__":
    main()
