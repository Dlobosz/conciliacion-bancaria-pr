"""Test de integracion del pipeline completo (sin llamar a la API real)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ia import AnalizadorExcepciones
from src.persistencia import conectar, leer_ejecucion
from src.pipeline import ejecutar

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(autouse=True)
def datos_disponibles():
    if not (RAW / "cartola_banco.csv").exists():
        pytest.skip("Faltan los datos sinteticos: correr python -m src.generar_datos")


def test_pipeline_completo_guarda_en_sqlite(tmp_path):
    ejecucion = ejecutar(
        RAW / "cartola_banco.csv", RAW / "libro_ventas.csv", ruta_db=tmp_path / "test.db"
    )

    assert ejecucion.id_ejecucion == 1
    assert not ejecucion.resultado.conciliaciones.empty

    conexion = conectar(tmp_path / "test.db")
    datos = leer_ejecucion(conexion, ejecucion.id_ejecucion)
    conexion.close()

    assert len(datos["movimientos"]) == len(ejecucion.movimientos)
    assert len(datos["conciliaciones"]) == len(ejecucion.resultado.conciliaciones)


def test_pipeline_sin_base_de_datos_igual_concilia():
    ejecucion = ejecutar(RAW / "cartola_banco.csv", RAW / "libro_ventas.csv")

    assert ejecucion.id_ejecucion is None
    assert not ejecucion.resultado.conciliaciones.empty


def test_sin_api_key_la_conciliacion_deterministica_igual_se_completa(tmp_path, monkeypatch):
    """La IA es un extra: si no esta configurada, el resultado deterministico no se pierde."""
    monkeypatch.setattr(AnalizadorExcepciones, "_obtener_api_key", lambda self: None)

    ejecucion = ejecutar(
        RAW / "cartola_banco.csv",
        RAW / "libro_ventas.csv",
        ruta_db=tmp_path / "test.db",
        usar_ia=True,
    )

    assert not ejecucion.resultado.conciliaciones.empty
    assert ejecucion.sugerencias_ia.empty
    assert any("ANTHROPIC_API_KEY" in error for error in ejecucion.errores_ia)
