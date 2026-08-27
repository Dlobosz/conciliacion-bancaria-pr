"""Lectura de archivos de entrada: cartola bancaria y libro de ventas (DTE).

Responsabilidad unica: leer el archivo (CSV o Excel) y verificar que traiga las
columnas minimas. Aqui no se normaliza nada; de eso se encarga limpieza.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columnas minimas que debe traer cada archivo para que el pipeline funcione.
COLUMNAS_CARTOLA = {"fecha", "descripcion", "monto"}
COLUMNAS_VENTAS = {"fecha_emision", "rut_cliente", "monto_total"}


class ArchivoInvalidoError(ValueError):
    """El archivo no existe, no tiene un formato soportado o le faltan columnas."""


def leer_tabla(ruta: str | Path) -> pd.DataFrame:
    """Lee un CSV o Excel a DataFrame, sin interpretar tipos (todo queda como texto).

    Se lee como texto a proposito: los montos y fechas vienen en formatos chilenos
    ("$1.234.567", "31-06-2025") que pandas interpretaria mal por su cuenta.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise ArchivoInvalidoError(f"No existe el archivo: {ruta}")

    if ruta.suffix.lower() == ".csv":
        return pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if ruta.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(ruta, dtype=str, keep_default_na=False)
    raise ArchivoInvalidoError(f"Formato no soportado: {ruta.suffix} (se admite .csv, .xlsx, .xls)")


def _normalizar_encabezados(df: pd.DataFrame) -> pd.DataFrame:
    """Encabezados a minusculas, sin espacios al borde y con guion bajo en vez de espacio."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _validar_columnas(df: pd.DataFrame, requeridas: set[str], origen: str) -> None:
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ArchivoInvalidoError(
            f"{origen}: faltan columnas obligatorias {sorted(faltantes)}. "
            f"Columnas encontradas: {sorted(df.columns)}"
        )


def leer_cartola(ruta: str | Path) -> pd.DataFrame:
    """Lee la cartola bancaria y valida que traiga fecha, descripcion y monto."""
    df = _normalizar_encabezados(leer_tabla(ruta))
    _validar_columnas(df, COLUMNAS_CARTOLA, "Cartola bancaria")
    return df


def leer_libro_ventas(ruta: str | Path) -> pd.DataFrame:
    """Lee el libro de ventas / registro de DTE y valida sus columnas minimas."""
    df = _normalizar_encabezados(leer_tabla(ruta))
    _validar_columnas(df, COLUMNAS_VENTAS, "Libro de ventas")
    return df
