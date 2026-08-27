"""Tests de normalizacion: montos, fechas, RUT y glosas."""

from __future__ import annotations

import pandas as pd
import pytest

from src.limpieza import (
    digito_verificador,
    extraer_contraparte,
    extraer_rut,
    ids_unicos,
    limpiar_cartola,
    limpiar_libro_ventas,
    normalizar_fecha,
    normalizar_monto,
    normalizar_rut,
    normalizar_texto,
    rut_valido,
)


@pytest.mark.parametrize(
    "crudo, esperado",
    [
        ("1234567", 1_234_567.0),
        ("1.234.567", 1_234_567.0),
        ("$1.234.567", 1_234_567.0),
        ("$ 1.234.567 ", 1_234_567.0),
        ("-9.900", -9_900.0),
        ("(9.900)", -9_900.0),
        ("1.234.567,89", 1_234_567.89),
        ("1234,5", 1_234.5),
        ("0", 0.0),
        ("", 0.0),
        (None, 0.0),
        (45000, 45_000.0),
    ],
)
def test_normalizar_monto(crudo, esperado):
    assert normalizar_monto(crudo) == pytest.approx(esperado)


def test_normalizar_monto_ignora_texto_basura():
    assert normalizar_monto("sin monto") == 0.0


@pytest.mark.parametrize(
    "crudo, esperado",
    [
        ("  Transf. Comercial  Ñuñoa S.A. ", "TRANSF COMERCIAL NUNOA S A"),
        ("PAGO N° 45", "PAGO N 45"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalizar_texto(crudo, esperado):
    assert normalizar_texto(crudo) == esperado


@pytest.mark.parametrize(
    "crudo, esperado",
    [
        ("76.123.456-0", "76123456-0"),
        ("76123456-0", "76123456-0"),
        ("76.123.456 - 0", "76123456-0"),
        ("12.345.678-k", "12345678-K"),
        ("123", ""),
        (None, ""),
    ],
)
def test_normalizar_rut(crudo, esperado):
    assert normalizar_rut(crudo) == esperado


def test_digito_verificador_conocido():
    assert digito_verificador("76123456") == "0"
    assert digito_verificador("77234567") == "4"


def test_rut_valido_detecta_dv_incorrecto():
    assert rut_valido("76.123.456-0")
    assert not rut_valido("76.123.456-9")


def test_extraer_rut_desde_glosa():
    glosa = "TRANSFERENCIA DE COMERCIAL LOS ALERCES SPA RUT 76.123.456-0"
    assert extraer_rut(glosa) == "76123456-0"


def test_extraer_rut_ignora_numero_que_no_es_rut():
    # Un numero de operacion con DV que no cuadra no debe usarse para cruzar.
    assert extraer_rut("ABONO OPERACION 12345678-9") == ""


def test_extraer_rut_sin_rut_devuelve_vacio():
    assert extraer_rut("COMISION MANTENCION CUENTA CORRIENTE") == ""


@pytest.mark.parametrize(
    "glosa, esperado",
    [
        (
            "TRANSFERENCIA DE COMERCIAL LOS ALERCES SPA RUT 76.123.456-0",
            "COMERCIAL LOS ALERCES",
        ),
        ("PAGO CONSOLIDADO DISTRIBUIDORA ANDINA LIMITADA", "DISTRIBUIDORA ANDINA"),
        ("ABONO CUOTA 2 CLINICA DENTAL SONRISA SPA", "CLINICA DENTAL SONRISA"),
        ("TRANSF SUPERMERCADOS EL PARRON S.A.", "SUPERMERCADOS EL PARRON"),
        ("DEPOSITO EN EFECTIVO CAJA SUCURSAL", "EN EFECTIVO CAJA SUCURSAL"),
    ],
)
def test_extraer_contraparte(glosa, esperado):
    assert extraer_contraparte(glosa) == esperado


def test_normalizar_fecha_acepta_iso_y_chileno():
    serie = pd.Series(["2025-06-15", "15-06-2025", "15/06/2025", "no es fecha"])
    fechas = normalizar_fecha(serie)
    assert fechas.iloc[0] == pd.Timestamp("2025-06-15")
    assert fechas.iloc[1] == pd.Timestamp("2025-06-15")
    assert fechas.iloc[2] == pd.Timestamp("2025-06-15")
    assert pd.isna(fechas.iloc[3])


def test_limpiar_cartola_columnas_y_signo():
    crudo = pd.DataFrame(
        {
            "id_movimiento": ["MOV-0001", "MOV-0002"],
            "fecha": ["2025-06-03", "2025-06-04"],
            "descripcion": [
                "TRANSFERENCIA DE COMERCIAL LOS ALERCES SPA RUT 76.123.456-0",
                "COMISION MANTENCION CUENTA CORRIENTE",
            ],
            "tipo": ["ABONO", "CARGO"],
            "monto": ["$1.500.000", "9.900"],
        }
    )
    limpio = limpiar_cartola(crudo)

    assert limpio.loc[0, "monto"] == 1_500_000.0
    assert limpio.loc[1, "monto"] == -9_900.0  # el tipo CARGO fuerza el signo negativo
    assert limpio.loc[1, "monto_abs"] == 9_900.0
    assert limpio.loc[0, "rut"] == "76123456-0"
    assert limpio.loc[0, "contraparte"] == "COMERCIAL LOS ALERCES"
    assert limpio.loc[1, "rut"] == ""


def test_limpiar_cartola_descarta_filas_sin_fecha_valida():
    crudo = pd.DataFrame(
        {
            "fecha": ["2025-06-03", "total del mes"],
            "descripcion": ["TRANSFERENCIA DE X", "TOTALES"],
            "monto": ["1000", "1000"],
        }
    )
    assert len(limpiar_cartola(crudo)) == 1


def test_limpiar_libro_ventas():
    crudo = pd.DataFrame(
        {
            "id_documento": ["DOC-1001"],
            "tipo_dte": ["33"],
            "folio": ["1001"],
            "fecha_emision": ["01-06-2025"],
            "rut_cliente": ["76.123.456-0"],
            "razon_social": ["Comercial Los Alerces SpA"],
            "monto_total": ["$1.500.000"],
        }
    )
    limpio = limpiar_libro_ventas(crudo)

    assert limpio.loc[0, "rut"] == "76123456-0"
    assert bool(limpio.loc[0, "rut_valido"])
    assert limpio.loc[0, "monto_total"] == 1_500_000.0
    assert limpio.loc[0, "contraparte"] == "COMERCIAL LOS ALERCES"
    assert limpio.loc[0, "fecha_emision"] == pd.Timestamp("2025-06-01")


# ------------------------------------------------- regresiones de la auditoria


def test_tipo_vacio_no_convierte_un_cargo_en_abono():
    """Regresion: con la columna 'tipo' en blanco el signo del monto debe respetarse.

    Antes se forzaba a positivo, y una comision bancaria terminaba compitiendo
    por conciliarse contra una factura de venta.
    """
    crudo = pd.DataFrame(
        {
            "fecha": ["2025-06-03", "2025-06-04"],
            "descripcion": ["CARGO SIN TIPO", "OTRO CARGO"],
            "tipo": ["", "DESCONOCIDO"],
            "monto": ["-45000", "-9900"],
        }
    )
    limpio = limpiar_cartola(crudo)

    assert limpio["monto"].tolist() == [-45_000.0, -9_900.0]
    assert limpio["tipo"].tolist() == ["CARGO", "CARGO"]


def test_tipo_explicito_manda_sobre_el_signo():
    crudo = pd.DataFrame(
        {
            "fecha": ["2025-06-03"],
            "descripcion": ["COMISION"],
            "tipo": ["CARGO"],
            "monto": ["9900"],  # el banco lo exporta sin signo
        }
    )
    assert limpiar_cartola(crudo).loc[0, "monto"] == -9_900.0


@pytest.mark.parametrize(
    "crudos, esperados",
    [
        (["MOV-1", "MOV-1"], ["MOV-1", "MOV-1-2"]),
        (["", ""], ["MOV-0001", "MOV-0002"]),
        (["MOV-1", "MOV-1", " MOV-1 "], ["MOV-1", "MOV-1-2", "MOV-1-3"]),
        (["MOV-1", "MOV-1-2", "MOV-1"], ["MOV-1", "MOV-1-2", "MOV-1-3"]),
    ],
)
def test_ids_unicos(crudos, esperados):
    assert ids_unicos(pd.Series(crudos), "MOV", len(crudos)) == esperados


def test_ids_duplicados_no_hacen_desaparecer_un_movimiento():
    """Regresion: dos filas con el mismo id colapsaban y una quedaba fuera del informe."""
    crudo = pd.DataFrame(
        {
            "id_movimiento": ["MOV-1", "MOV-1", ""],
            "fecha": ["2025-06-03"] * 3,
            "descripcion": ["A", "B", "C"],
            "tipo": ["ABONO"] * 3,
            "monto": ["100", "200", "300"],
        }
    )
    limpio = limpiar_cartola(crudo)

    assert len(limpio) == 3
    assert limpio["id_movimiento"].is_unique


def test_los_documentos_tambien_tienen_id_unico():
    crudo = pd.DataFrame(
        {
            "id_documento": ["DOC-1", "DOC-1"],
            "fecha_emision": ["2025-06-03", "2025-06-04"],
            "rut_cliente": ["76.123.456-0"] * 2,
            "monto_total": ["100", "200"],
        }
    )
    assert limpiar_libro_ventas(crudo)["id_documento"].is_unique
