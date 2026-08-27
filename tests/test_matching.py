"""Tests del motor de conciliacion deterministico.

Es la logica critica del proyecto: cada caso dificil (comision, 1-a-N, N-a-1,
duplicados, glosas ambiguas) tiene su test, y el resultado debe ser identico
entre ejecuciones.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.matching import ParametrosMatching, conciliar

FECHA = pd.Timestamp("2025-06-10")


def mov(id_mov, monto, dia=0, rut="76123456-0", contraparte="COMERCIAL LOS ALERCES"):
    """Un movimiento ya limpio (como lo entrega limpieza.limpiar_cartola)."""
    return {
        "id_movimiento": id_mov,
        "fecha": FECHA + pd.Timedelta(days=dia),
        "descripcion": f"TRANSFERENCIA DE {contraparte}",
        "glosa": f"TRANSFERENCIA DE {contraparte}",
        "monto": float(monto),
        "tipo": "CARGO" if monto < 0 else "ABONO",
        "monto_abs": abs(float(monto)),
        "rut": rut,
        "contraparte": contraparte,
    }


def doc(id_doc, monto, dia=0, rut="76123456-0", contraparte="COMERCIAL LOS ALERCES"):
    """Un documento ya limpio (como lo entrega limpieza.limpiar_libro_ventas)."""
    return {
        "id_documento": id_doc,
        "folio": id_doc.split("-")[-1],
        "tipo_dte": "33",
        "fecha_emision": FECHA + pd.Timedelta(days=dia),
        "rut": rut,
        "rut_valido": True,
        "razon_social": contraparte,
        "contraparte": contraparte,
        "monto_total": float(monto),
    }


def correr(movimientos, documentos, params=None):
    return conciliar(pd.DataFrame(movimientos), pd.DataFrame(documentos), params)


def pares(resultado) -> set[tuple[str, str]]:
    """Expande las conciliaciones a pares (movimiento, documento) para comparar."""
    encontrados = set()
    for fila in resultado.conciliaciones.itertuples():
        for m in fila.ids_movimientos.split("|"):
            for d in fila.ids_documentos.split("|"):
                encontrados.add((m, d))
    return encontrados


# ---------------------------------------------------------------- match exacto


def test_match_exacto():
    r = correr([mov("MOV-1", 1_500_000)], [doc("DOC-1", 1_500_000)])

    assert len(r.conciliaciones) == 1
    fila = r.conciliaciones.iloc[0]
    assert fila["estrategia"] == "exacto"
    assert fila["confianza"] == 1.0
    assert not fila["requiere_revision"]
    assert fila["diferencia"] == 0
    assert r.movimientos_pendientes.empty
    assert r.documentos_pendientes.empty


def test_no_cruza_por_monto_si_el_rut_es_distinto():
    r = correr(
        [mov("MOV-1", 1_500_000, rut="76123456-0")],
        [doc("DOC-1", 1_500_000, rut="77234567-4", contraparte="OTRA EMPRESA")],
    )
    assert r.conciliaciones.empty
    assert len(r.movimientos_pendientes) == 1
    assert len(r.documentos_pendientes) == 1


def test_pago_duplicado_solo_concilia_uno():
    """El segundo abono identico no puede consumir el mismo documento."""
    r = correr(
        [mov("MOV-1", 800_000), mov("MOV-2", 800_000, dia=2)],
        [doc("DOC-1", 800_000)],
    )

    assert len(r.conciliaciones) == 1
    assert len(r.movimientos_pendientes) == 1
    assert r.movimientos_pendientes.iloc[0]["motivo"] == "posible_pago_duplicado"


# ---------------------------------------------------------- tolerancia de fecha


@pytest.mark.parametrize("desfase", [1, 3, 5])
def test_tolerancia_fecha_dentro_de_la_ventana(desfase):
    r = correr([mov("MOV-1", 950_000, dia=desfase)], [doc("DOC-1", 950_000)])

    assert len(r.conciliaciones) == 1
    assert r.conciliaciones.iloc[0]["estrategia"] == "tolerancia_fecha"


def test_tolerancia_fecha_fuera_de_la_ventana_no_concilia():
    r = correr([mov("MOV-1", 950_000, dia=9)], [doc("DOC-1", 950_000)])
    assert r.conciliaciones.empty


def test_ventana_de_fecha_es_configurable():
    params = ParametrosMatching(ventana_dias=10)
    r = correr([mov("MOV-1", 950_000, dia=9)], [doc("DOC-1", 950_000)], params)
    assert len(r.conciliaciones) == 1


# ------------------------------------------------------------ comision bancaria


def test_comision_bancaria_deja_la_diferencia_registrada():
    r = correr([mov("MOV-1", 1_000_000 - 3_500, dia=1)], [doc("DOC-1", 1_000_000)])

    fila = r.conciliaciones.iloc[0]
    assert fila["estrategia"] == "comision"
    assert fila["diferencia"] == 3_500
    assert "comision" in fila["detalle"].lower()


def test_comision_mayor_al_tope_no_concilia():
    # 1% de 1.000.000 = 10.000; una diferencia de 40.000 no es una comision.
    r = correr([mov("MOV-1", 960_000, dia=1)], [doc("DOC-1", 1_000_000)])
    assert r.conciliaciones.empty


def test_abono_mayor_al_documento_no_es_comision():
    """Si entra MAS plata que la facturada, no es comision: es otra cosa y debe revisarse."""
    r = correr([mov("MOV-1", 1_003_500, dia=1)], [doc("DOC-1", 1_000_000)])
    assert r.conciliaciones.empty


# ------------------------------------------------------------- fuzzy por nombre


def test_fuzzy_concilia_glosa_truncada_sin_rut():
    r = correr(
        [mov("MOV-1", 640_000, rut="", contraparte="DISTRIBUIDORA ANDINA")],
        [doc("DOC-1", 640_000, contraparte="DISTRIBUIDORA ANDINA LIMITADA")],
    )

    fila = r.conciliaciones.iloc[0]
    assert fila["estrategia"] == "fuzzy"
    assert 0.6 <= fila["confianza"] <= 1.0


def test_fuzzy_rechaza_nombre_muy_distinto():
    r = correr(
        [mov("MOV-1", 640_000, rut="", contraparte="PANADERIA DON LUCHO")],
        [doc("DOC-1", 640_000, contraparte="TRANSPORTES CORDILLERA")],
    )
    assert r.conciliaciones.empty
    assert r.movimientos_pendientes.iloc[0]["motivo"] == "sin_documento_calzado"


def test_fuzzy_no_concilia_si_el_monto_no_calza():
    """El nombre nunca manda sobre la plata: el monto es condicion dura."""
    r = correr(
        [mov("MOV-1", 300_000, rut="", contraparte="DISTRIBUIDORA ANDINA")],
        [doc("DOC-1", 640_000, contraparte="DISTRIBUIDORA ANDINA LIMITADA")],
    )
    assert r.conciliaciones.empty


def test_fuzzy_con_empate_deja_el_caso_pendiente():
    """Dos documentos igual de parecidos: el motor no adivina, lo manda a revision."""
    r = correr(
        [mov("MOV-1", 500_000, rut="", contraparte="COMERCIAL LOS ALERCES")],
        [
            doc("DOC-1", 500_000, contraparte="COMERCIAL LOS ALERCES"),
            doc("DOC-2", 500_000, rut="77234567-4", contraparte="COMERCIAL LOS ALERCES"),
        ],
    )
    assert r.conciliaciones.empty
    assert len(r.movimientos_pendientes) == 1
    assert r.movimientos_pendientes.iloc[0]["candidatos"] != ""


# ----------------------------------------------------------------- N-a-N


def test_uno_a_n_un_abono_paga_tres_facturas():
    r = correr(
        [mov("MOV-1", 300_000 + 450_000 + 120_000, dia=2)],
        [doc("DOC-1", 300_000), doc("DOC-2", 450_000, dia=1), doc("DOC-3", 120_000, dia=1)],
    )

    fila = r.conciliaciones.iloc[0]
    assert fila["estrategia"] == "uno_a_n"
    assert fila["n_documentos"] == 3
    assert fila["diferencia"] == 0
    assert r.documentos_pendientes.empty


def test_uno_a_n_ignora_documentos_de_otro_cliente():
    """La combinatoria se acota por RUT: no puede 'cuadrar' mezclando clientes."""
    r = correr(
        [mov("MOV-1", 750_000, dia=2)],
        [doc("DOC-1", 300_000), doc("DOC-2", 450_000, rut="77234567-4", contraparte="OTRA")],
    )
    assert r.conciliaciones.empty


def test_n_a_1_tres_cuotas_pagan_una_factura():
    r = correr(
        [mov("MOV-1", 500_000, dia=1), mov("MOV-2", 300_000, dia=8), mov("MOV-3", 200_000, dia=15)],
        [doc("DOC-1", 1_000_000)],
    )

    fila = r.conciliaciones.iloc[0]
    assert fila["estrategia"] == "n_a_1"
    assert fila["n_movimientos"] == 3
    assert fila["monto_banco"] == 1_000_000


# ----------------------------------------------------------------- pendientes


def test_los_cargos_nunca_se_concilian_contra_ventas():
    r = correr(
        [mov("MOV-1", -9_900, contraparte="COMISION MANTENCION")],
        [doc("DOC-1", 9_900)],
    )
    assert r.conciliaciones.empty
    assert r.movimientos_pendientes.iloc[0]["motivo"] == "cargo_sin_documento"
    assert len(r.documentos_pendientes) == 1


def test_documento_sin_pago_queda_pendiente():
    r = correr([], [doc("DOC-1", 1_200_000)])
    assert r.documentos_pendientes.iloc[0]["motivo"] == "sin_pago_registrado"


def test_entrada_vacia_no_revienta():
    r = correr([], [])
    assert r.conciliaciones.empty
    assert r.movimientos_pendientes.empty
    assert r.documentos_pendientes.empty


# ----------------------------------------------------------------- determinismo


def test_el_resultado_es_reproducible():
    """Misma entrada, mismo resultado: es el argumento central del diseno."""
    movimientos = [mov("MOV-1", 500_000), mov("MOV-2", 300_000, dia=1)]
    documentos = [doc("DOC-1", 500_000), doc("DOC-2", 300_000, dia=1)]

    primera = correr(movimientos, documentos)
    segunda = correr(movimientos, documentos)

    pd.testing.assert_frame_equal(primera.conciliaciones, segunda.conciliaciones)
    assert pares(primera) == pares(segunda)


# ------------------------------------------------- integracion con datos reales


@pytest.fixture(scope="module")
def resultado_dataset():
    """Corre el pipeline completo sobre los CSV sinteticos de data/raw."""
    from pathlib import Path

    from src.ingesta import leer_cartola, leer_libro_ventas
    from src.limpieza import limpiar_cartola, limpiar_libro_ventas

    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    if not (raw / "ground_truth.csv").exists():
        pytest.skip("Faltan los datos sinteticos: correr python -m src.generar_datos")

    movimientos = limpiar_cartola(leer_cartola(raw / "cartola_banco.csv"))
    documentos = limpiar_libro_ventas(leer_libro_ventas(raw / "libro_ventas.csv"))
    verdad = pd.read_csv(raw / "ground_truth.csv", keep_default_na=False)
    return conciliar(movimientos, documentos), verdad


def test_dataset_sin_falsos_positivos(resultado_dataset):
    """Precision 100%: el motor puede dejar casos pendientes, pero no inventar matches."""
    resultado, verdad = resultado_dataset
    esperados = {(f.id_movimiento, f.id_documento) for f in verdad.itertuples() if f.id_documento}
    propuestos = pares(resultado)

    assert propuestos - esperados == set()


def test_dataset_cobertura_minima(resultado_dataset):
    resultado, verdad = resultado_dataset
    esperados = {
        (f.id_movimiento, f.id_documento)
        for f in verdad.itertuples()
        if f.id_documento and f.id_movimiento
    }
    cobertura = len(pares(resultado) & esperados) / len(esperados)

    assert cobertura >= 0.90


def test_dataset_detecta_los_duplicados_y_los_cargos(resultado_dataset):
    resultado, _ = resultado_dataset
    motivos = resultado.movimientos_pendientes["motivo"].value_counts()

    assert motivos.get("posible_pago_duplicado", 0) == 2
    assert motivos.get("cargo_sin_documento", 0) == 4


def test_dataset_facturas_impagas_quedan_pendientes(resultado_dataset):
    resultado, verdad = resultado_dataset
    impagas = {f.id_documento for f in verdad.itertuples() if f.caso == "sin_pago"}

    assert impagas <= set(resultado.documentos_pendientes["id_documento"])
