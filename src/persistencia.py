"""Persistencia en SQLite: guarda y recupera el resultado de cada conciliacion.

Se eligio SQLite y no PostgreSQL porque la aplicacion corre en un solo proceso,
sin concurrencia real, y asi no hay que administrar un servidor. Si el proyecto
escalara a multiusuario, el esquema migra a Postgres sin cambios de modelo.

Modelo de datos:

    ejecuciones          una corrida del pipeline (que archivos, cuando, resumen)
    movimientos          cartola bancaria limpia de esa ejecucion
    documentos           libro de ventas limpio de esa ejecucion
    conciliaciones       cada match propuesto, con estrategia y confianza
    conciliacion_items   que movimientos y documentos entran en cada match (N a N)
    pendientes           lo que quedo sin conciliar, de ambos lados

Todas las tablas cuelgan de id_ejecucion: se puede volver a ver una conciliacion
anterior sin perder la trazabilidad.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.matching import ResultadoConciliacion

DB_POR_DEFECTO = Path("data") / "conciliacion.db"

ESQUEMA = """
CREATE TABLE IF NOT EXISTS ejecuciones (
    id_ejecucion      INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha_ejecucion   TEXT    NOT NULL,
    archivo_cartola   TEXT,
    archivo_ventas    TEXT,
    n_movimientos     INTEGER NOT NULL,
    n_documentos      INTEGER NOT NULL,
    n_conciliaciones  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS movimientos (
    id_ejecucion   INTEGER NOT NULL REFERENCES ejecuciones(id_ejecucion) ON DELETE CASCADE,
    id_movimiento  TEXT    NOT NULL,
    fecha          TEXT    NOT NULL,
    descripcion    TEXT,
    tipo           TEXT,
    monto          REAL    NOT NULL,
    rut            TEXT,
    contraparte    TEXT,
    PRIMARY KEY (id_ejecucion, id_movimiento)
);

CREATE TABLE IF NOT EXISTS documentos (
    id_ejecucion   INTEGER NOT NULL REFERENCES ejecuciones(id_ejecucion) ON DELETE CASCADE,
    id_documento   TEXT    NOT NULL,
    folio          TEXT,
    tipo_dte       TEXT,
    fecha_emision  TEXT    NOT NULL,
    rut            TEXT,
    razon_social   TEXT,
    monto_total    REAL    NOT NULL,
    PRIMARY KEY (id_ejecucion, id_documento)
);

CREATE TABLE IF NOT EXISTS conciliaciones (
    id_ejecucion      INTEGER NOT NULL REFERENCES ejecuciones(id_ejecucion) ON DELETE CASCADE,
    id_conciliacion   TEXT    NOT NULL,
    estrategia        TEXT    NOT NULL,
    confianza         REAL    NOT NULL,
    requiere_revision INTEGER NOT NULL,
    n_movimientos     INTEGER NOT NULL,
    n_documentos      INTEGER NOT NULL,
    monto_banco       REAL    NOT NULL,
    monto_documentos  REAL    NOT NULL,
    diferencia        REAL    NOT NULL,
    detalle           TEXT,
    PRIMARY KEY (id_ejecucion, id_conciliacion)
);

CREATE TABLE IF NOT EXISTS conciliacion_items (
    id_ejecucion    INTEGER NOT NULL REFERENCES ejecuciones(id_ejecucion) ON DELETE CASCADE,
    id_conciliacion TEXT    NOT NULL,
    lado            TEXT    NOT NULL CHECK (lado IN ('movimiento', 'documento')),
    id_item         TEXT    NOT NULL,
    PRIMARY KEY (id_ejecucion, id_conciliacion, lado, id_item)
);

CREATE TABLE IF NOT EXISTS pendientes (
    id_ejecucion    INTEGER NOT NULL REFERENCES ejecuciones(id_ejecucion) ON DELETE CASCADE,
    lado            TEXT    NOT NULL CHECK (lado IN ('movimiento', 'documento')),
    id_item         TEXT    NOT NULL,
    fecha           TEXT,
    descripcion     TEXT,
    monto           REAL,
    rut             TEXT,
    motivo          TEXT    NOT NULL,
    candidatos      TEXT,
    clasificacion_ia TEXT,
    sugerencia_ia   TEXT,
    confianza_ia    REAL,
    explicacion_ia  TEXT,
    PRIMARY KEY (id_ejecucion, lado, id_item)
);

CREATE INDEX IF NOT EXISTS idx_items_ejecucion ON conciliacion_items (id_ejecucion, id_item);
CREATE INDEX IF NOT EXISTS idx_pendientes_motivo ON pendientes (id_ejecucion, motivo);
"""


def conectar(ruta_db: str | Path = DB_POR_DEFECTO) -> sqlite3.Connection:
    """Abre la base (creandola si no existe) con el esquema listo."""
    ruta_db = Path(ruta_db)
    if ruta_db.parent != Path("") and str(ruta_db) != ":memory:":
        ruta_db.parent.mkdir(parents=True, exist_ok=True)

    conexion = sqlite3.connect(ruta_db)
    conexion.execute("PRAGMA foreign_keys = ON")
    crear_esquema(conexion)
    return conexion


def crear_esquema(conexion: sqlite3.Connection) -> None:
    conexion.executescript(ESQUEMA)
    conexion.commit()


def _fechas_a_texto(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """SQLite no tiene tipo fecha: se guarda ISO (yyyy-mm-dd), que ordena bien como texto."""
    copia = df.copy()
    for columna in columnas:
        if columna in copia.columns:
            copia[columna] = pd.to_datetime(copia[columna], errors="coerce").dt.strftime("%Y-%m-%d")
    return copia


def _insertar(conexion: sqlite3.Connection, tabla: str, df: pd.DataFrame, columnas: list[str]) -> None:
    if df.empty:
        return
    df.reindex(columns=columnas).to_sql(tabla, conexion, if_exists="append", index=False)


def _items_de_conciliaciones(conciliaciones: pd.DataFrame, id_ejecucion: int) -> pd.DataFrame:
    """Aplana ids_movimientos / ids_documentos ('A|B') a filas de la tabla puente."""
    filas = []
    for fila in conciliaciones.itertuples():
        for lado, ids in (("movimiento", fila.ids_movimientos), ("documento", fila.ids_documentos)):
            for id_item in filter(None, str(ids).split("|")):
                filas.append(
                    {
                        "id_ejecucion": id_ejecucion,
                        "id_conciliacion": fila.id_conciliacion,
                        "lado": lado,
                        "id_item": id_item,
                    }
                )
    return pd.DataFrame(filas)


def guardar_ejecucion(
    conexion: sqlite3.Connection,
    movimientos: pd.DataFrame,
    documentos: pd.DataFrame,
    resultado: ResultadoConciliacion,
    archivo_cartola: str = "",
    archivo_ventas: str = "",
) -> int:
    """Guarda una corrida completa y devuelve su id_ejecucion."""
    cursor = conexion.execute(
        "INSERT INTO ejecuciones (fecha_ejecucion, archivo_cartola, archivo_ventas, "
        "n_movimientos, n_documentos, n_conciliaciones) VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.now().isoformat(timespec="seconds"),
            archivo_cartola,
            archivo_ventas,
            len(movimientos),
            len(documentos),
            len(resultado.conciliaciones),
        ),
    )
    id_ejecucion = int(cursor.lastrowid)

    movs = _fechas_a_texto(movimientos, ["fecha"]).assign(id_ejecucion=id_ejecucion)
    _insertar(
        conexion,
        "movimientos",
        movs,
        ["id_ejecucion", "id_movimiento", "fecha", "descripcion", "tipo", "monto", "rut", "contraparte"],
    )

    docs = _fechas_a_texto(documentos, ["fecha_emision"]).assign(id_ejecucion=id_ejecucion)
    _insertar(
        conexion,
        "documentos",
        docs,
        ["id_ejecucion", "id_documento", "folio", "tipo_dte", "fecha_emision", "rut", "razon_social", "monto_total"],
    )

    conciliaciones = resultado.conciliaciones.assign(id_ejecucion=id_ejecucion)
    if not conciliaciones.empty:
        conciliaciones["requiere_revision"] = conciliaciones["requiere_revision"].astype(int)
    _insertar(
        conexion,
        "conciliaciones",
        conciliaciones,
        [
            "id_ejecucion",
            "id_conciliacion",
            "estrategia",
            "confianza",
            "requiere_revision",
            "n_movimientos",
            "n_documentos",
            "monto_banco",
            "monto_documentos",
            "diferencia",
            "detalle",
        ],
    )
    _insertar(
        conexion,
        "conciliacion_items",
        _items_de_conciliaciones(resultado.conciliaciones, id_ejecucion),
        ["id_ejecucion", "id_conciliacion", "lado", "id_item"],
    )

    pendientes_mov = (
        _fechas_a_texto(resultado.movimientos_pendientes, ["fecha"])
        .rename(columns={"id_movimiento": "id_item"})
        .assign(id_ejecucion=id_ejecucion, lado="movimiento")
    )
    pendientes_doc = (
        _fechas_a_texto(resultado.documentos_pendientes, ["fecha_emision"])
        .rename(
            columns={
                "id_documento": "id_item",
                "fecha_emision": "fecha",
                "razon_social": "descripcion",
                "monto_total": "monto",
            }
        )
        .assign(id_ejecucion=id_ejecucion, lado="documento")
    )
    columnas_pendientes = [
        "id_ejecucion",
        "lado",
        "id_item",
        "fecha",
        "descripcion",
        "monto",
        "rut",
        "motivo",
        "candidatos",
    ]
    _insertar(conexion, "pendientes", pendientes_mov, columnas_pendientes)
    _insertar(conexion, "pendientes", pendientes_doc, columnas_pendientes)

    conexion.commit()
    return id_ejecucion


def ultima_ejecucion(conexion: sqlite3.Connection) -> int | None:
    fila = conexion.execute("SELECT MAX(id_ejecucion) FROM ejecuciones").fetchone()
    return int(fila[0]) if fila and fila[0] is not None else None


def leer_tabla(conexion: sqlite3.Connection, tabla: str, id_ejecucion: int) -> pd.DataFrame:
    """Lee una tabla filtrada por ejecucion (las fechas vuelven como datetime)."""
    df = pd.read_sql_query(
        f"SELECT * FROM {tabla} WHERE id_ejecucion = ?", conexion, params=(id_ejecucion,)
    )
    for columna in ("fecha", "fecha_emision"):
        if columna in df.columns:
            df[columna] = pd.to_datetime(df[columna], errors="coerce")
    return df


def leer_ejecucion(conexion: sqlite3.Connection, id_ejecucion: int | None = None) -> dict:
    """Devuelve todas las tablas de una ejecucion (por defecto, la ultima)."""
    id_ejecucion = id_ejecucion or ultima_ejecucion(conexion)
    if id_ejecucion is None:
        raise ValueError("No hay ninguna ejecucion guardada en la base de datos")

    return {
        "id_ejecucion": id_ejecucion,
        "movimientos": leer_tabla(conexion, "movimientos", id_ejecucion),
        "documentos": leer_tabla(conexion, "documentos", id_ejecucion),
        "conciliaciones": leer_tabla(conexion, "conciliaciones", id_ejecucion),
        "conciliacion_items": leer_tabla(conexion, "conciliacion_items", id_ejecucion),
        "pendientes": leer_tabla(conexion, "pendientes", id_ejecucion),
    }


def guardar_sugerencia_ia(
    conexion: sqlite3.Connection,
    id_ejecucion: int,
    id_item: str,
    clasificacion: str,
    sugerencia: str,
    confianza: float,
    explicacion: str,
    lado: str = "movimiento",
) -> None:
    """Anota lo que propuso el LLM sobre un pendiente.

    La sugerencia se guarda aparte del resultado deterministico y NO lo modifica:
    queda esperando la revision humana.
    """
    conexion.execute(
        "UPDATE pendientes SET clasificacion_ia = ?, sugerencia_ia = ?, confianza_ia = ?, "
        "explicacion_ia = ? WHERE id_ejecucion = ? AND lado = ? AND id_item = ?",
        (clasificacion, sugerencia, confianza, explicacion, id_ejecucion, lado, id_item),
    )
    conexion.commit()
