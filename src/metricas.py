"""Metricas del proceso: cuanto se concilio solo, cuanto quedo pendiente y cuanto tiempo se ahorra.

Sirven para dos cosas: los KPI del dashboard y las cifras honestas del README.
Todos los supuestos estan explicitos como constantes para poder discutirlos.
"""

from __future__ import annotations

import pandas as pd

from src.matching import UMBRAL_AUTOMATICO, ResultadoConciliacion

# Supuesto de tiempo: conciliar un movimiento a mano (buscarlo en el libro de
# ventas, verificar monto y fecha, marcarlo) toma del orden de 2 minutos. Con
# 200-600 movimientos al mes eso da las 4-20 h/mes que reporta el sector.
MINUTOS_POR_MOVIMIENTO_MANUAL = 2.0

# Revisar un caso que el sistema dejo pendiente igual cuesta tiempo, pero menos:
# viene acotado, clasificado y con candidatos.
MINUTOS_POR_PENDIENTE_REVISADO = 1.0


def _ids_conciliados(conciliaciones: pd.DataFrame, columna: str) -> set[str]:
    if conciliaciones.empty:
        return set()
    ids: set[str] = set()
    for valor in conciliaciones[columna]:
        ids.update(filter(None, str(valor).split("|")))
    return ids


def _porcentaje(parte: float, total: float) -> float:
    return round(100 * parte / total, 1) if total else 0.0


def calcular(
    movimientos: pd.DataFrame,
    documentos: pd.DataFrame,
    resultado: ResultadoConciliacion,
    revision: dict | None = None,
) -> dict:
    """Resumen cuantitativo de una conciliacion.

    `revision` es el dict que devuelve persistencia.resumen_revision(): permite
    separar lo que resolvio el motor de lo que confirmo una persona.
    """
    conciliaciones = resultado.conciliaciones

    # Los cargos (comisiones, remuneraciones, impuestos) no se cruzan contra ventas:
    # medir la cobertura sobre ellos castigaria el resultado sin sentido contable.
    abonos = movimientos[movimientos["monto"] > 0] if not movimientos.empty else movimientos

    movs_conciliados = _ids_conciliados(conciliaciones, "ids_movimientos")
    docs_conciliados = _ids_conciliados(conciliaciones, "ids_documentos")

    automaticas = (
        conciliaciones[conciliaciones["confianza"] >= UMBRAL_AUTOMATICO]
        if not conciliaciones.empty
        else conciliaciones
    )

    # Lo que una persona ya reviso: confirmado a mano o descartado con motivo.
    por_lado = (revision or {}).get("movimiento", {})
    conciliados_manual = por_lado.get("conciliado_manual", 0)
    descartados = por_lado.get("descartado", 0)
    sin_revisar = max(len(resultado.movimientos_pendientes) - conciliados_manual - descartados, 0)

    minutos_manual = len(movimientos) * MINUTOS_POR_MOVIMIENTO_MANUAL
    minutos_con_sistema = len(resultado.movimientos_pendientes) * MINUTOS_POR_PENDIENTE_REVISADO

    return {
        "conciliados_manualmente": conciliados_manual,
        "descartados_en_revision": descartados,
        "pendientes_sin_revisar": sin_revisar,
        "pct_revisado": _porcentaje(
            conciliados_manual + descartados, len(resultado.movimientos_pendientes)
        ),
        "pct_abonos_cerrados": _porcentaje(
            len(_ids_conciliados(conciliaciones, "ids_movimientos")) + conciliados_manual,
            len(abonos),
        ),
        "total_movimientos": len(movimientos),
        "total_abonos": len(abonos),
        "total_documentos": len(documentos),
        "n_conciliaciones": len(conciliaciones),
        "movimientos_conciliados": len(movs_conciliados),
        "documentos_conciliados": len(docs_conciliados),
        "pct_abonos_conciliados": _porcentaje(len(movs_conciliados), len(abonos)),
        "pct_documentos_conciliados": _porcentaje(len(docs_conciliados), len(documentos)),
        "conciliaciones_automaticas": len(automaticas),
        "pct_automatico": _porcentaje(len(automaticas), len(conciliaciones)),
        "movimientos_pendientes": len(resultado.movimientos_pendientes),
        "documentos_pendientes": len(resultado.documentos_pendientes),
        "monto_conciliado": (
            float(conciliaciones["monto_banco"].sum()) if not conciliaciones.empty else 0.0
        ),
        "monto_pendiente": float(resultado.movimientos_pendientes["monto"].abs().sum())
        if not resultado.movimientos_pendientes.empty
        else 0.0,
        "diferencias_por_comision": float(conciliaciones["diferencia"].sum())
        if not conciliaciones.empty
        else 0.0,
        "por_estrategia": (
            conciliaciones["estrategia"].value_counts().to_dict()
            if not conciliaciones.empty
            else {}
        ),
        "por_motivo_pendiente": (
            resultado.movimientos_pendientes["motivo"].value_counts().to_dict()
            if not resultado.movimientos_pendientes.empty
            else {}
        ),
        "horas_manual": round(minutos_manual / 60, 1),
        "horas_con_sistema": round(minutos_con_sistema / 60, 1),
        "horas_ahorradas": round((minutos_manual - minutos_con_sistema) / 60, 1),
    }


def precision_contra_verdad(resultado: ResultadoConciliacion, verdad: pd.DataFrame) -> dict:
    """Precision y cobertura contra un set etiquetado (data/raw/ground_truth.csv).

    precision = matches correctos / matches propuestos  -> cuantos de los que propuso estaban bien
    cobertura = matches correctos / matches esperados   -> cuantos de los que existian encontro

    En conciliacion bancaria la precision importa mas que la cobertura: un match
    incorrecto ensucia la contabilidad, mientras que uno que falta solo queda
    pendiente de revision.
    """
    propuestos: set[tuple[str, str]] = set()
    for fila in resultado.conciliaciones.itertuples():
        for id_mov in filter(None, str(fila.ids_movimientos).split("|")):
            for id_doc in filter(None, str(fila.ids_documentos).split("|")):
                propuestos.add((id_mov, id_doc))

    esperados = {
        (str(fila.id_movimiento), str(fila.id_documento))
        for fila in verdad.itertuples()
        if str(fila.id_movimiento) and str(fila.id_documento)
    }
    correctos = propuestos & esperados

    return {
        "propuestos": len(propuestos),
        "esperados": len(esperados),
        "correctos": len(correctos),
        "falsos_positivos": sorted(propuestos - esperados),
        "no_encontrados": sorted(esperados - propuestos),
        "precision": round(len(correctos) / len(propuestos), 4) if propuestos else 0.0,
        "cobertura": round(len(correctos) / len(esperados), 4) if esperados else 0.0,
    }


def resumen_texto(metricas: dict) -> str:
    """Resumen en una linea para la consola o el README."""
    return (
        f"{metricas['movimientos_conciliados']}/{metricas['total_abonos']} abonos conciliados "
        f"({metricas['pct_abonos_conciliados']}%), "
        f"{metricas['movimientos_pendientes']} pendientes de revision, "
        f"~{metricas['horas_ahorradas']} h ahorradas en el mes"
    )
