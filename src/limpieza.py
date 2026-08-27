"""Normalizacion de los datos crudos antes del matching.

Todo lo que el motor de conciliacion asume ya resuelto vive aqui: montos en
numero, fechas en datetime, RUT en un formato unico y glosas comparables
(mayusculas, sin tildes ni puntuacion). Es codigo deterministico y testeable.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# RUT escrito de cualquier forma: 76.123.456-0 / 76123456-0 / 76.123.456 - K
PATRON_RUT = re.compile(r"(\d{1,3}(?:\.\d{3})+|\d{6,9})\s*-\s*([\dkK])")

# Ruido tipico al inicio de una glosa bancaria, antes del nombre de la contraparte.
PREFIJOS_GLOSA = re.compile(
    r"^(TRANSFERENCIA DE|TRANSFERENCIA|TRANSF|PAGO CONSOLIDADO|PAGO DE|PAGO|"
    r"ABONO CUOTA \d+|ABONO DE|ABONO|DEPOSITO DE|DEPOSITO|RECIBIDO DE|CARGO)\s+"
)

# Sufijos societarios y letras sueltas que quedan al normalizar "S.A." o "E.I.R.L.":
# no aportan a la comparacion y solo castigan el score fuzzy.
TOKENS_RUIDO = {"SPA", "SA", "LTDA", "LIMITADA", "EIRL", "S", "A", "E", "I", "R", "L", "CIA"}


def quitar_sufijos_societarios(texto: str) -> str:
    """'COMERCIAL LOS ALERCES SPA' -> 'COMERCIAL LOS ALERCES'."""
    palabras = [p for p in texto.split() if p not in TOKENS_RUIDO and not p.isdigit()]
    return " ".join(palabras)


def sin_tildes(texto: str) -> str:
    """ACENTÚA -> ACENTUA (descompone en NFD y bota las marcas diacriticas)."""
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def normalizar_texto(valor) -> str:
    """Deja una glosa lista para comparar: mayusculas, sin tildes ni puntuacion.

    >>> normalizar_texto("  Transf. Comercial  Ñuñoa S.A. ")
    'TRANSF COMERCIAL NUNOA S A'
    """
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return ""
    texto = sin_tildes(str(valor)).upper()
    texto = re.sub(r"[^A-Z0-9\s-]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def normalizar_monto(valor) -> float:
    """Convierte '$1.234.567', '1.234.567,89' o '(9.900)' a float.

    Regla chilena: el punto es separador de miles y la coma de decimales.
    Los parentesis indican monto negativo (convencion contable).
    """
    if valor is None or valor == "" or (isinstance(valor, float) and pd.isna(valor)):
        return 0.0
    if isinstance(valor, (int, float)):
        return float(valor)

    texto = str(valor).strip()
    negativo = texto.startswith("(") and texto.endswith(")")
    texto = re.sub(r"[^\d,.\-]", "", texto)
    if not texto or texto in {"-", ".", ","}:
        return 0.0

    if "," in texto:  # la coma manda como decimal: los puntos son miles
        texto = texto.replace(".", "").replace(",", ".")
    elif texto.count(".") == 1 and len(texto.split(".")[-1]) != 3:
        pass  # un solo punto con 1, 2 o 4+ decimales: ya es punto decimal
    else:
        texto = texto.replace(".", "")

    try:
        monto = float(texto)
    except ValueError:
        return 0.0
    return -abs(monto) if negativo else monto


def digito_verificador(cuerpo: str) -> str:
    """DV de un RUT chileno (modulo 11)."""
    suma, factor = 0, 2
    for digito in reversed(cuerpo):
        suma += int(digito) * factor
        factor = 2 if factor == 7 else factor + 1
    resto = 11 - (suma % 11)
    return {11: "0", 10: "K"}.get(resto, str(resto))


def normalizar_rut(valor) -> str:
    """'76.123.456-0' -> '76123456-0'. Devuelve '' si no hay un RUT reconocible."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return ""
    crudo = re.sub(r"[^\dkK]", "", str(valor)).upper()
    if len(crudo) < 7:
        return ""
    return f"{crudo[:-1]}-{crudo[-1]}"


def rut_valido(valor) -> bool:
    """Verifica el digito verificador. Un RUT mal digitado no debe usarse para cruzar."""
    rut = normalizar_rut(valor)
    if not rut:
        return False
    cuerpo, dv = rut.split("-")
    return cuerpo.isdigit() and digito_verificador(cuerpo) == dv


def extraer_rut(glosa) -> str:
    """Saca el RUT de una glosa bancaria ('TRANSF X RUT 76.123.456-0' -> '76123456-0').

    Solo devuelve el RUT si el digito verificador cuadra: asi evitamos cruzar por
    un numero de operacion que casualmente parezca RUT.
    """
    if glosa is None:
        return ""
    for cuerpo, dv in PATRON_RUT.findall(str(glosa)):
        candidato = normalizar_rut(f"{cuerpo}-{dv}")
        if rut_valido(candidato):
            return candidato
    return ""


def extraer_contraparte(glosa) -> str:
    """Deja solo el nombre probable de la contraparte, sin ruido ni RUT.

    'TRANSFERENCIA DE COMERCIAL LOS ALERCES SPA RUT 76.123.456-0'
    -> 'COMERCIAL LOS ALERCES'
    """
    # El RUT se quita ANTES de normalizar: despues los puntos ya no estarian.
    texto = PATRON_RUT.sub(" ", str(glosa or ""))
    texto = normalizar_texto(texto)
    texto = re.sub(r"\bRUT\b", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    anterior = None
    while anterior != texto:  # puede haber prefijos encadenados: "PAGO TRANSF X"
        anterior = texto
        texto = PREFIJOS_GLOSA.sub("", texto).strip()

    return quitar_sufijos_societarios(texto)


def normalizar_fecha(serie: pd.Series) -> pd.Series:
    """Convierte una columna de fechas a datetime, tolerando dd-mm-yyyy y yyyy-mm-dd."""
    texto = serie.astype(str).str.strip()
    iso = texto.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
    fechas = pd.to_datetime(texto, errors="coerce", format="mixed", dayfirst=True)
    fechas_iso = pd.to_datetime(texto.where(iso), errors="coerce", format="mixed", dayfirst=False)
    return fechas_iso.fillna(fechas).dt.normalize()


def limpiar_cartola(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza la cartola bancaria y deja las columnas que consume el matching."""
    limpio = pd.DataFrame(index=df.index)

    limpio["id_movimiento"] = (
        df["id_movimiento"].astype(str)
        if "id_movimiento" in df.columns
        else [f"MOV-{i + 1:04d}" for i in range(len(df))]
    )
    limpio["fecha"] = normalizar_fecha(df["fecha"])
    limpio["descripcion"] = df["descripcion"].astype(str).str.strip()
    limpio["glosa"] = df["descripcion"].map(normalizar_texto)
    limpio["monto"] = df["monto"].map(normalizar_monto)

    # El signo manda sobre la columna 'tipo': un CARGO siempre resta.
    if "tipo" in df.columns:
        tipo = df["tipo"].map(normalizar_texto)
        limpio["monto"] = limpio["monto"].abs() * tipo.map(lambda t: -1 if t == "CARGO" else 1)
    limpio["tipo"] = limpio["monto"].map(lambda m: "CARGO" if m < 0 else "ABONO")
    limpio["monto_abs"] = limpio["monto"].abs()

    limpio["rut"] = df["descripcion"].map(extraer_rut)
    limpio["contraparte"] = df["descripcion"].map(extraer_contraparte)

    return limpio.dropna(subset=["fecha"]).reset_index(drop=True)


def limpiar_libro_ventas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el libro de ventas / registro de DTE."""
    limpio = pd.DataFrame(index=df.index)

    folios = df["folio"].astype(str) if "folio" in df.columns else pd.Series(
        [str(i + 1) for i in range(len(df))], index=df.index
    )
    limpio["id_documento"] = (
        df["id_documento"].astype(str) if "id_documento" in df.columns else "DOC-" + folios
    )
    limpio["folio"] = folios
    limpio["tipo_dte"] = df["tipo_dte"].astype(str) if "tipo_dte" in df.columns else "33"
    limpio["fecha_emision"] = normalizar_fecha(df["fecha_emision"])
    limpio["rut"] = df["rut_cliente"].map(normalizar_rut)
    limpio["rut_valido"] = limpio["rut"].map(rut_valido)
    limpio["razon_social"] = df["razon_social"] if "razon_social" in df.columns else ""
    limpio["contraparte"] = limpio["razon_social"].map(
        lambda r: quitar_sufijos_societarios(normalizar_texto(r))
    )
    limpio["monto_total"] = df["monto_total"].map(normalizar_monto).abs()

    return limpio.dropna(subset=["fecha_emision"]).reset_index(drop=True)
