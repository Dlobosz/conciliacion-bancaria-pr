"""Tests de las metricas del proceso."""

from __future__ import annotations

import pandas as pd
import pytest

from src.matching import conciliar
from src.metricas import calcular, precision_contra_verdad, resumen_texto
from tests.test_matching import doc, mov


@pytest.fixture
def caso():
    """2 abonos conciliados, 1 cargo pendiente y 1 factura impaga."""
    movimientos = pd.DataFrame(
        [
            mov("MOV-1", 500_000),
            mov("MOV-2", 300_000, dia=1, rut="77234567-4", contraparte="ANDINA"),
            mov("MOV-3", -9_900, contraparte="COMISION MANTENCION"),
        ]
    )
    documentos = pd.DataFrame(
        [
            doc("DOC-1", 500_000),
            doc("DOC-2", 300_000, dia=1, rut="77234567-4", contraparte="ANDINA"),
            doc("DOC-3", 750_000, rut="78567890-7", contraparte="VALLE VERDE"),
        ]
    )
    return movimientos, documentos, conciliar(movimientos, documentos)


def test_conteos_basicos(caso):
    m = calcular(*caso)

    assert m["total_movimientos"] == 3
    assert m["total_abonos"] == 2  # el cargo no cuenta para la cobertura
    assert m["movimientos_conciliados"] == 2
    assert m["pct_abonos_conciliados"] == 100.0
    assert m["documentos_pendientes"] == 1


def test_los_cargos_no_castigan_la_cobertura(caso):
    """El % se mide sobre abonos: un cargo bancario nunca deberia calzar con una venta."""
    m = calcular(*caso)
    assert m["pct_abonos_conciliados"] == 100.0
    assert m["movimientos_pendientes"] == 1


def test_montos_y_estrategias(caso):
    m = calcular(*caso)

    assert m["monto_conciliado"] == 800_000
    assert m["monto_pendiente"] == 9_900
    assert m["por_estrategia"]["exacto"] == 2
    assert m["por_motivo_pendiente"]["cargo_sin_documento"] == 1


def test_tiempo_ahorrado(caso):
    m = calcular(*caso)

    # 3 movimientos x 2 min manual = 6 min; 1 pendiente x 1 min = 1 min
    assert m["horas_manual"] == round(6 / 60, 1)
    assert m["horas_ahorradas"] == round(5 / 60, 1)


def test_metricas_con_todo_vacio():
    vacio = pd.DataFrame()
    m = calcular(vacio, vacio, conciliar(vacio, vacio))

    assert m["pct_abonos_conciliados"] == 0.0
    assert m["horas_ahorradas"] == 0.0


def test_precision_perfecta(caso):
    _, _, resultado = caso
    verdad = pd.DataFrame(
        [
            {"id_movimiento": "MOV-1", "id_documento": "DOC-1", "caso": "exacto"},
            {"id_movimiento": "MOV-2", "id_documento": "DOC-2", "caso": "exacto"},
            {"id_movimiento": "", "id_documento": "DOC-3", "caso": "sin_pago"},
        ]
    )
    p = precision_contra_verdad(resultado, verdad)

    assert p["precision"] == 1.0
    assert p["cobertura"] == 1.0
    assert p["falsos_positivos"] == []


def test_precision_detecta_un_match_faltante(caso):
    _, _, resultado = caso
    verdad = pd.DataFrame(
        [
            {"id_movimiento": "MOV-1", "id_documento": "DOC-1", "caso": "exacto"},
            {"id_movimiento": "MOV-2", "id_documento": "DOC-2", "caso": "exacto"},
            {"id_movimiento": "MOV-3", "id_documento": "DOC-3", "caso": "inventado"},
        ]
    )
    p = precision_contra_verdad(resultado, verdad)

    assert p["precision"] == 1.0  # no propuso nada incorrecto
    assert p["cobertura"] == round(2 / 3, 4)
    assert p["no_encontrados"] == [("MOV-3", "DOC-3")]


def test_resumen_texto(caso):
    texto = resumen_texto(calcular(*caso))
    assert "2/2 abonos conciliados" in texto
    assert "pendientes de revision" in texto


# --------------------------------------------------------------- revision humana


def test_sin_revision_los_contadores_van_en_cero(caso):
    m = calcular(*caso)

    assert m["conciliados_manualmente"] == 0
    assert m["pendientes_sin_revisar"] == m["movimientos_pendientes"]
    assert m["pct_revisado"] == 0.0
    assert m["pct_abonos_cerrados"] == m["pct_abonos_conciliados"]


def test_la_revision_humana_suma_a_los_abonos_cerrados(caso):
    movimientos, documentos, resultado = caso
    revision = {"movimiento": {"conciliado_manual": 1}, "documento": {"conciliado_manual": 1}}

    m = calcular(movimientos, documentos, resultado, revision)

    assert m["conciliados_manualmente"] == 1
    assert m["pendientes_sin_revisar"] == 0
    assert m["pct_revisado"] == 100.0


def test_un_descarte_tambien_cuenta_como_revisado(caso):
    movimientos, documentos, resultado = caso
    revision = {"movimiento": {"descartado": 1}}

    m = calcular(movimientos, documentos, resultado, revision)

    assert m["descartados_en_revision"] == 1
    assert m["pendientes_sin_revisar"] == 0
    # descartar no concilia nada: el % de abonos cerrados no se mueve
    assert m["pct_abonos_cerrados"] == m["pct_abonos_conciliados"]
