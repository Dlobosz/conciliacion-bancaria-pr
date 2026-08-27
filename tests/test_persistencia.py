"""Tests de persistencia: guardar y recuperar una conciliacion completa."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src.matching import conciliar
from src.persistencia import (
    conectar,
    guardar_ejecucion,
    guardar_sugerencia_ia,
    leer_ejecucion,
    ultima_ejecucion,
)
from tests.test_matching import doc, mov


@pytest.fixture
def conexion(tmp_path):
    con = conectar(tmp_path / "test.db")
    yield con
    con.close()


@pytest.fixture
def caso():
    """Un caso con un match exacto, un 1-a-N, un cargo pendiente y una factura impaga."""
    movimientos = pd.DataFrame(
        [
            mov("MOV-1", 500_000),
            mov("MOV-2", 300_000 + 200_000, dia=2, rut="77234567-4", contraparte="ANDINA"),
            mov("MOV-3", -9_900, contraparte="COMISION MANTENCION"),
        ]
    )
    documentos = pd.DataFrame(
        [
            doc("DOC-1", 500_000),
            doc("DOC-2", 300_000, dia=1, rut="77234567-4", contraparte="ANDINA"),
            doc("DOC-3", 200_000, dia=1, rut="77234567-4", contraparte="ANDINA"),
            doc("DOC-4", 750_000, rut="78567890-7", contraparte="VALLE VERDE"),
        ]
    )
    return movimientos, documentos, conciliar(movimientos, documentos)


def test_crear_esquema_es_idempotente(tmp_path):
    ruta = tmp_path / "test.db"
    conectar(ruta).close()
    con = conectar(ruta)  # abrir de nuevo no debe fallar ni duplicar tablas
    tablas = {f[0] for f in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()

    assert {
        "ejecuciones",
        "movimientos",
        "documentos",
        "conciliaciones",
        "conciliacion_items",
        "pendientes",
    } <= tablas


def test_guardar_y_leer_ejecucion(conexion, caso):
    movimientos, documentos, resultado = caso
    id_ejecucion = guardar_ejecucion(
        conexion, movimientos, documentos, resultado, "cartola.csv", "ventas.csv"
    )

    datos = leer_ejecucion(conexion, id_ejecucion)
    assert len(datos["movimientos"]) == 3
    assert len(datos["documentos"]) == 4
    assert len(datos["conciliaciones"]) == len(resultado.conciliaciones)
    assert datos["movimientos"]["fecha"].dtype.kind == "M"  # vuelve como datetime


def test_los_items_reconstruyen_el_match_n_a_n(conexion, caso):
    movimientos, documentos, resultado = caso
    id_ejecucion = guardar_ejecucion(conexion, movimientos, documentos, resultado)

    items = leer_ejecucion(conexion, id_ejecucion)["conciliacion_items"]
    uno_a_n = resultado.conciliaciones[resultado.conciliaciones["estrategia"] == "uno_a_n"]
    id_conciliacion = uno_a_n.iloc[0]["id_conciliacion"]

    documentos_del_match = set(
        items[(items["id_conciliacion"] == id_conciliacion) & (items["lado"] == "documento")][
            "id_item"
        ]
    )
    assert documentos_del_match == {"DOC-2", "DOC-3"}


def test_pendientes_guardan_ambos_lados(conexion, caso):
    movimientos, documentos, resultado = caso
    id_ejecucion = guardar_ejecucion(conexion, movimientos, documentos, resultado)

    pendientes = leer_ejecucion(conexion, id_ejecucion)["pendientes"]
    assert set(pendientes["lado"]) == {"movimiento", "documento"}
    assert "MOV-3" in set(pendientes["id_item"])  # el cargo
    assert "DOC-4" in set(pendientes["id_item"])  # la factura impaga


def test_varias_ejecuciones_no_se_mezclan(conexion, caso):
    movimientos, documentos, resultado = caso
    primera = guardar_ejecucion(conexion, movimientos, documentos, resultado)
    segunda = guardar_ejecucion(conexion, movimientos.head(1), documentos.head(1), resultado)

    assert segunda == primera + 1
    assert ultima_ejecucion(conexion) == segunda
    assert len(leer_ejecucion(conexion, primera)["movimientos"]) == 3
    assert len(leer_ejecucion(conexion, segunda)["movimientos"]) == 1


def test_leer_ejecucion_sin_datos_falla_claro(conexion):
    with pytest.raises(ValueError, match="No hay ninguna ejecucion"):
        leer_ejecucion(conexion)


def test_guardar_sugerencia_ia_no_toca_lo_deterministico(conexion, caso):
    movimientos, documentos, resultado = caso
    id_ejecucion = guardar_ejecucion(conexion, movimientos, documentos, resultado)
    conciliaciones_antes = leer_ejecucion(conexion, id_ejecucion)["conciliaciones"]

    guardar_sugerencia_ia(
        conexion,
        id_ejecucion,
        "MOV-3",
        clasificacion="comision_bancaria",
        sugerencia="",
        confianza=0.92,
        explicacion="Cargo de mantencion de cuenta, no corresponde a una venta",
    )

    datos = leer_ejecucion(conexion, id_ejecucion)
    pendiente = datos["pendientes"].set_index("id_item").loc["MOV-3"]
    assert pendiente["clasificacion_ia"] == "comision_bancaria"
    assert pendiente["confianza_ia"] == 0.92
    pd.testing.assert_frame_equal(conciliaciones_antes, datos["conciliaciones"])


def test_borrar_ejecucion_arrastra_sus_tablas(conexion, caso):
    movimientos, documentos, resultado = caso
    id_ejecucion = guardar_ejecucion(conexion, movimientos, documentos, resultado)

    conexion.execute("DELETE FROM ejecuciones WHERE id_ejecucion = ?", (id_ejecucion,))
    conexion.commit()

    restantes = conexion.execute(
        "SELECT COUNT(*) FROM movimientos WHERE id_ejecucion = ?", (id_ejecucion,)
    ).fetchone()[0]
    assert restantes == 0


def test_no_se_puede_guardar_un_lado_invalido(conexion, caso):
    movimientos, documentos, resultado = caso
    id_ejecucion = guardar_ejecucion(conexion, movimientos, documentos, resultado)

    with pytest.raises(sqlite3.IntegrityError):
        conexion.execute(
            "INSERT INTO pendientes (id_ejecucion, lado, id_item, motivo) VALUES (?, ?, ?, ?)",
            (id_ejecucion, "otro", "X", "motivo"),
        )
