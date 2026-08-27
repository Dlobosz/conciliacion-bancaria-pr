"""Tests de persistencia: guardar y recuperar una conciliacion completa."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src.matching import conciliar
from src.persistencia import (
    RevisionInvalidaError,
    conciliar_manualmente,
    conectar,
    descartar_pendiente,
    guardar_ejecucion,
    guardar_sugerencia_ia,
    leer_conciliaciones,
    leer_ejecucion,
    reabrir_pendiente,
    resumen_revision,
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


# --------------------------------------------------------------- revision humana


@pytest.fixture
def ejecucion_guardada(conexion, caso):
    movimientos, documentos, resultado = caso
    return conexion, guardar_ejecucion(conexion, movimientos, documentos, resultado)


def test_conciliar_manualmente_crea_una_conciliacion(ejecucion_guardada):
    conexion, id_ejecucion = ejecucion_guardada

    id_conciliacion = conciliar_manualmente(
        conexion, id_ejecucion, "MOV-3", "DOC-4", comentario="Es el pago de esa factura"
    )

    datos = leer_ejecucion(conexion, id_ejecucion)
    manual = datos["conciliaciones"].set_index("id_conciliacion").loc[id_conciliacion]
    assert manual["estrategia"] == "revision_humana"
    assert manual["confianza"] == 1.0
    assert not manual["requiere_revision"]
    assert "Es el pago de esa factura" in manual["detalle"]

    items = datos["conciliacion_items"]
    del_match = items[items["id_conciliacion"] == id_conciliacion]
    assert set(del_match["id_item"]) == {"MOV-3", "DOC-4"}


def test_conciliar_manualmente_marca_los_dos_lados(ejecucion_guardada):
    conexion, id_ejecucion = ejecucion_guardada
    conciliar_manualmente(conexion, id_ejecucion, "MOV-3", "DOC-4")

    pendientes = leer_ejecucion(conexion, id_ejecucion)["pendientes"].set_index("id_item")
    assert pendientes.loc["MOV-3", "estado_revision"] == "conciliado_manual"
    assert pendientes.loc["DOC-4", "estado_revision"] == "conciliado_manual"
    assert pendientes.loc["MOV-3", "resuelto_con"] == "DOC-4"


def test_no_se_puede_resolver_dos_veces_el_mismo_pendiente(ejecucion_guardada):
    conexion, id_ejecucion = ejecucion_guardada
    conciliar_manualmente(conexion, id_ejecucion, "MOV-3", "DOC-4")

    with pytest.raises(RevisionInvalidaError, match="ya fue resuelto"):
        conciliar_manualmente(conexion, id_ejecucion, "MOV-3", "DOC-4")


def test_no_se_puede_conciliar_un_item_que_no_esta_pendiente(ejecucion_guardada):
    """MOV-1 ya lo concilio el motor: la revision no puede volver a usarlo."""
    conexion, id_ejecucion = ejecucion_guardada

    with pytest.raises(RevisionInvalidaError, match="no figura como pendiente"):
        conciliar_manualmente(conexion, id_ejecucion, "MOV-1", "DOC-4")


def test_descartar_pendiente_lo_cierra_sin_documento(ejecucion_guardada):
    conexion, id_ejecucion = ejecucion_guardada

    descartar_pendiente(conexion, id_ejecucion, "MOV-3", "Comision de mantencion, no es una venta")

    datos = leer_ejecucion(conexion, id_ejecucion)
    pendiente = datos["pendientes"].set_index("id_item").loc["MOV-3"]
    assert pendiente["estado_revision"] == "descartado"
    assert "Comision" in pendiente["comentario_revision"]
    assert datos["conciliaciones"]["estrategia"].tolist().count("revision_humana") == 0


def test_descartar_exige_un_motivo(ejecucion_guardada):
    conexion, id_ejecucion = ejecucion_guardada
    with pytest.raises(RevisionInvalidaError, match="por que se descarta"):
        descartar_pendiente(conexion, id_ejecucion, "MOV-3", "   ")


def test_reabrir_deshace_la_conciliacion_manual(ejecucion_guardada):
    conexion, id_ejecucion = ejecucion_guardada
    conciliaciones_antes = len(leer_ejecucion(conexion, id_ejecucion)["conciliaciones"])
    conciliar_manualmente(conexion, id_ejecucion, "MOV-3", "DOC-4")

    reabrir_pendiente(conexion, id_ejecucion, "MOV-3")

    datos = leer_ejecucion(conexion, id_ejecucion)
    assert len(datos["conciliaciones"]) == conciliaciones_antes
    pendientes = datos["pendientes"].set_index("id_item")
    assert pendientes.loc["MOV-3", "estado_revision"] == "pendiente"
    assert pendientes.loc["DOC-4", "estado_revision"] == "pendiente"  # se reabren los dos lados


def test_reabrir_un_descarte(ejecucion_guardada):
    conexion, id_ejecucion = ejecucion_guardada
    descartar_pendiente(conexion, id_ejecucion, "MOV-3", "no corresponde")

    reabrir_pendiente(conexion, id_ejecucion, "MOV-3")

    pendientes = leer_ejecucion(conexion, id_ejecucion)["pendientes"].set_index("id_item")
    assert pendientes.loc["MOV-3", "estado_revision"] == "pendiente"


def test_resumen_revision(ejecucion_guardada):
    conexion, id_ejecucion = ejecucion_guardada
    conciliar_manualmente(conexion, id_ejecucion, "MOV-3", "DOC-4")

    resumen = resumen_revision(conexion, id_ejecucion)
    assert resumen["movimiento"]["conciliado_manual"] == 1
    assert resumen["documento"]["conciliado_manual"] == 1


def test_todo_parte_como_pendiente(ejecucion_guardada):
    conexion, id_ejecucion = ejecucion_guardada
    resumen = resumen_revision(conexion, id_ejecucion)

    assert "conciliado_manual" not in resumen["movimiento"]
    assert resumen["movimiento"]["pendiente"] >= 1


def test_una_base_del_esquema_viejo_se_migra(tmp_path):
    """Una base creada antes de la revision humana no se pierde: se le agregan las columnas."""
    ruta = tmp_path / "vieja.db"
    vieja = sqlite3.connect(ruta)
    vieja.execute(
        "CREATE TABLE pendientes (id_ejecucion INTEGER, lado TEXT, id_item TEXT, motivo TEXT)"
    )
    vieja.execute("INSERT INTO pendientes VALUES (1, 'movimiento', 'MOV-9', 'sin_documento')")
    vieja.commit()
    vieja.close()

    conexion = conectar(ruta)
    columnas = {fila[1] for fila in conexion.execute("PRAGMA table_info(pendientes)")}
    estado = conexion.execute("SELECT estado_revision FROM pendientes").fetchone()[0]
    conexion.close()

    assert {"estado_revision", "resuelto_con", "id_conciliacion_manual"} <= columnas
    assert estado == "pendiente"


def test_leer_conciliaciones_reconstruye_los_ids(ejecucion_guardada, caso):
    """Lo guardado y lo calculado en memoria deben tener la misma forma."""
    conexion, id_ejecucion = ejecucion_guardada
    _, _, resultado = caso

    guardadas = leer_conciliaciones(conexion, id_ejecucion).set_index("id_conciliacion")
    en_memoria = resultado.conciliaciones.set_index("id_conciliacion")

    for id_conciliacion, fila in en_memoria.iterrows():
        assert guardadas.loc[id_conciliacion, "ids_movimientos"] == "|".join(
            sorted(fila["ids_movimientos"].split("|"))
        )
        assert guardadas.loc[id_conciliacion, "ids_documentos"] == "|".join(
            sorted(fila["ids_documentos"].split("|"))
        )
    assert guardadas["requiere_revision"].dtype == bool


def test_leer_conciliaciones_incluye_las_manuales(ejecucion_guardada):
    conexion, id_ejecucion = ejecucion_guardada
    antes = len(leer_conciliaciones(conexion, id_ejecucion))

    id_conciliacion = conciliar_manualmente(conexion, id_ejecucion, "MOV-3", "DOC-4")

    despues = leer_conciliaciones(conexion, id_ejecucion).set_index("id_conciliacion")
    assert len(despues) == antes + 1
    assert despues.loc[id_conciliacion, "ids_movimientos"] == "MOV-3"
    assert despues.loc[id_conciliacion, "ids_documentos"] == "DOC-4"


def test_una_cartola_con_ids_repetidos_se_guarda_completa(conexion):
    """Regresion: los ids duplicados rompian el guardado con UNIQUE constraint failed."""
    from src.limpieza import limpiar_cartola, limpiar_libro_ventas

    cartola = limpiar_cartola(
        pd.DataFrame(
            {
                "id_movimiento": ["MOV-1", "MOV-1", ""],
                "fecha": ["2025-06-03", "2025-06-04", "2025-06-05"],
                "descripcion": ["A", "B", "C"],
                "tipo": ["ABONO"] * 3,
                "monto": ["100", "200", "300"],
            }
        )
    )
    ventas = limpiar_libro_ventas(
        pd.DataFrame(
            columns=["id_documento", "fecha_emision", "rut_cliente", "razon_social", "monto_total"]
        )
    )

    id_ejecucion = guardar_ejecucion(conexion, cartola, ventas, conciliar(cartola, ventas))

    assert len(leer_ejecucion(conexion, id_ejecucion)["movimientos"]) == 3
