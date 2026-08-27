"""Smoke test del dashboard: que la app corra y muestre los KPI sin errores."""

from __future__ import annotations

from pathlib import Path

import pytest

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(autouse=True)
def datos_disponibles():
    if not (RAW / "cartola_banco.csv").exists():
        pytest.skip("Faltan los datos sinteticos: correr python -m src.generar_datos")


def test_la_app_carga_sin_excepciones():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    app.run()

    assert not app.exception
    assert app.title[0].value == "Conciliacion bancaria"
    assert len(app.metric) >= 5  # los 5 KPI de la cabecera
