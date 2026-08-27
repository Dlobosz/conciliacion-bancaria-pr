"""Tests de lectura y validacion de archivos de entrada."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingesta import ArchivoInvalidoError, leer_cartola, leer_libro_ventas, leer_tabla


@pytest.fixture
def csv_cartola(tmp_path):
    ruta = tmp_path / "cartola.csv"
    ruta.write_text(
        "Fecha,Descripcion,Tipo,Monto\n2025-06-01,TRANSFERENCIA DE X,ABONO,1500000\n",
        encoding="utf-8",
    )
    return ruta


def test_leer_cartola_normaliza_encabezados(csv_cartola):
    df = leer_cartola(csv_cartola)
    assert list(df.columns) == ["fecha", "descripcion", "tipo", "monto"]
    assert len(df) == 1


def test_leer_tabla_deja_todo_como_texto(csv_cartola):
    df = leer_tabla(csv_cartola)
    assert df["Monto"].iloc[0] == "1500000"  # sin conversion automatica de pandas


def test_archivo_inexistente(tmp_path):
    with pytest.raises(ArchivoInvalidoError, match="No existe"):
        leer_cartola(tmp_path / "no_existe.csv")


def test_formato_no_soportado(tmp_path):
    ruta = tmp_path / "cartola.txt"
    ruta.write_text("dato", encoding="utf-8")
    with pytest.raises(ArchivoInvalidoError, match="Formato no soportado"):
        leer_cartola(ruta)


def test_cartola_sin_columna_obligatoria(tmp_path):
    ruta = tmp_path / "mala.csv"
    ruta.write_text("fecha,glosa\n2025-06-01,ALGO\n", encoding="utf-8")
    with pytest.raises(ArchivoInvalidoError, match="faltan columnas"):
        leer_cartola(ruta)


def test_leer_excel(tmp_path):
    ruta = tmp_path / "ventas.xlsx"
    pd.DataFrame(
        {"fecha_emision": ["2025-06-01"], "rut_cliente": ["76.123.456-0"], "monto_total": [1500000]}
    ).to_excel(ruta, index=False)
    df = leer_libro_ventas(ruta)
    assert df["rut_cliente"].iloc[0] == "76.123.456-0"
