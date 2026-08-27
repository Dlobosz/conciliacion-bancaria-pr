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
    estado_revision TEXT NOT NULL DEFAULT 'pendiente'
        CHECK (estado_revision IN ('pendiente', 'conciliado_manual', 'descartado')),
    resuelto_con           TEXT,
    id_conciliacion_manual TEXT,
    comentario_revision    TEXT,
    fecha_revision         TEXT,
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


# Columnas de revision humana agregadas despues de la primera version del esquema.
# Se migran con ALTER TABLE para no perder las bases que ya existan.
COLUMNAS_REVISION = {
    "estado_revision": "TEXT NOT NULL DEFAULT 'pendiente'",
    "resuelto_con": "TEXT",
    "id_conciliacion_manual": "TEXT",
    "comentario_revision": "TEXT",
    "fecha_revision": "TEXT",
}


def crear_esquema(conexion: sqlite3.Connection) -> None:
    conexion.executescript(ESQUEMA)
    _migrar_pendientes(conexion)
    conexion.commit()


def _migrar_pendientes(conexion: sqlite3.Connection) -> None:
    """Agrega las columnas de revision a una base creada con el esquema anterior."""
    existentes = {fila[1] for fila in conexion.execute("PRAGMA table_info(pendientes)")}
    for columna, definicion in COLUMNAS_REVISION.items():
        if columna not in existentes:
            conexion.execute(f"ALTER TABLE pendientes ADD COLUMN {columna} {definicion}")


def _fechas_a_texto(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """SQLite no tiene tipo fecha: se guarda ISO (yyyy-mm-dd), que ordena bien como texto."""
    copia = df.copy()
    for columna in columnas:
        if columna in copia.columns:
            copia[columna] = pd.to_datetime(copia[columna], errors="coerce").dt.strftime("%Y-%m-%d")
    return copia


def _insertar(
    conexion: sqlite3.Connection, tabla: str, df: pd.DataFrame, columnas: list[str]
) -> None:
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
        [
            "id_ejecucion",
            "id_movimiento",
            "fecha",
            "descripcion",
            "tipo",
            "monto",
            "rut",
            "contraparte",
        ],
    )

    docs = _fechas_a_texto(documentos, ["fecha_emision"]).assign(id_ejecucion=id_ejecucion)
    _insertar(
        conexion,
        "documentos",
        docs,
        [
            "id_ejecucion",
            "id_documento",
            "folio",
            "tipo_dte",
            "fecha_emision",
            "rut",
            "razon_social",
            "monto_total",
        ],
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


def leer_conciliaciones(conexion: sqlite3.Connection, id_ejecucion: int) -> pd.DataFrame:
    """Conciliaciones de una ejecucion con la misma forma que devuelve el motor.

    Reconstruye ids_movimientos / ids_documentos desde la tabla puente, para que
    lo guardado y lo calculado en memoria se puedan usar indistintamente. Incluye
    las conciliaciones creadas en revision humana.
    """
    conciliaciones = leer_tabla(conexion, "conciliaciones", id_ejecucion)
    if conciliaciones.empty:
        return conciliaciones.assign(ids_movimientos="", ids_documentos="")

    items = leer_tabla(conexion, "conciliacion_items", id_ejecucion)
    agrupados = (
        items.pivot_table(
            index="id_conciliacion",
            columns="lado",
            values="id_item",
            aggfunc=lambda ids: "|".join(sorted(ids)),
            fill_value="",
        )
        .reindex(columns=["movimiento", "documento"], fill_value="")
    )

    conciliaciones = conciliaciones.merge(
        agrupados.rename(columns={"movimiento": "ids_movimientos", "documento": "ids_documentos"}),
        left_on="id_conciliacion",
        right_index=True,
        how="left",
    )
    conciliaciones[["ids_movimientos", "ids_documentos"]] = conciliaciones[
        ["ids_movimientos", "ids_documentos"]
    ].fillna("")
    conciliaciones["requiere_revision"] = conciliaciones["requiere_revision"].astype(bool)
    return conciliaciones


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


# --------------------------------------------------------------------------------------
# Revision humana
#
# Es el paso que cierra el proceso: el motor propone y la IA sugiere, pero un caso
# pendiente solo pasa a conciliado cuando una persona lo confirma. Lo que se decide
# aqui queda como una conciliacion mas, con estrategia 'revision_humana', de modo
# que las metricas y las descargas la incluyen sin logica aparte.
# --------------------------------------------------------------------------------------

ESTRATEGIA_MANUAL = "revision_humana"


class RevisionInvalidaError(ValueError):
    """La decision de revision no se puede aplicar (item inexistente, ya resuelto, doc ocupado)."""


def _pendiente(conexion: sqlite3.Connection, id_ejecucion: int, lado: str, id_item: str) -> dict:
    fila = conexion.execute(
        "SELECT * FROM pendientes WHERE id_ejecucion = ? AND lado = ? AND id_item = ?",
        (id_ejecucion, lado, id_item),
    ).fetchone()
    if fila is None:
        raise RevisionInvalidaError(
            f"{id_item} no figura como pendiente de la ejecucion {id_ejecucion}"
        )

    columnas = [d[0] for d in conexion.execute("SELECT * FROM pendientes LIMIT 0").description]
    pendiente = dict(zip(columnas, fila, strict=True))
    if pendiente["estado_revision"] != "pendiente":
        raise RevisionInvalidaError(
            f"{id_item} ya fue resuelto ({pendiente['estado_revision']}); "
            "reabrelo antes de cambiarlo"
        )
    return pendiente


def _monto_de(conexion: sqlite3.Connection, tabla: str, id_ejecucion: int, id_item: str) -> float:
    columna_id = "id_movimiento" if tabla == "movimientos" else "id_documento"
    columna_monto = "monto" if tabla == "movimientos" else "monto_total"
    fila = conexion.execute(
        f"SELECT {columna_monto} FROM {tabla} WHERE id_ejecucion = ? AND {columna_id} = ?",
        (id_ejecucion, id_item),
    ).fetchone()
    if fila is None:
        raise RevisionInvalidaError(f"{id_item} no existe en la ejecucion {id_ejecucion}")
    return abs(float(fila[0]))


def conciliar_manualmente(
    conexion: sqlite3.Connection,
    id_ejecucion: int,
    id_movimiento: str,
    id_documento: str,
    comentario: str = "",
    usuario: str = "revision_humana",
) -> str:
    """Confirma a mano que un movimiento pendiente corresponde a un documento.

    Se mantiene la misma invariante que el motor deterministico: ni el movimiento
    ni el documento pueden estar ya usados en otra conciliacion.
    """
    _pendiente(conexion, id_ejecucion, "movimiento", id_movimiento)
    _pendiente(conexion, id_ejecucion, "documento", id_documento)

    ocupado = conexion.execute(
        "SELECT id_conciliacion FROM conciliacion_items "
        "WHERE id_ejecucion = ? AND id_item IN (?, ?)",
        (id_ejecucion, id_movimiento, id_documento),
    ).fetchone()
    if ocupado:
        raise RevisionInvalidaError(
            f"{id_movimiento} o {id_documento} ya participan en la conciliacion {ocupado[0]}"
        )

    monto_banco = _monto_de(conexion, "movimientos", id_ejecucion, id_movimiento)
    monto_documento = _monto_de(conexion, "documentos", id_ejecucion, id_documento)

    correlativo = conexion.execute(
        "SELECT COUNT(*) FROM conciliaciones WHERE id_ejecucion = ? AND estrategia = ?",
        (id_ejecucion, ESTRATEGIA_MANUAL),
    ).fetchone()[0]
    id_conciliacion = f"CON-M{correlativo + 1:03d}"

    detalle = f"Confirmado en revision humana ({usuario})"
    if comentario:
        detalle = f"{detalle}: {comentario}"

    conexion.execute(
        "INSERT INTO conciliaciones (id_ejecucion, id_conciliacion, estrategia, confianza, "
        "requiere_revision, n_movimientos, n_documentos, monto_banco, monto_documentos, "
        "diferencia, detalle) VALUES (?, ?, ?, 1.0, 0, 1, 1, ?, ?, ?, ?)",
        (
            id_ejecucion,
            id_conciliacion,
            ESTRATEGIA_MANUAL,
            monto_banco,
            monto_documento,
            round(monto_documento - monto_banco, 2),
            detalle,
        ),
    )
    conexion.executemany(
        "INSERT INTO conciliacion_items (id_ejecucion, id_conciliacion, lado, id_item) "
        "VALUES (?, ?, ?, ?)",
        [
            (id_ejecucion, id_conciliacion, "movimiento", id_movimiento),
            (id_ejecucion, id_conciliacion, "documento", id_documento),
        ],
    )

    ahora = datetime.now().isoformat(timespec="seconds")
    for lado, id_item, contraparte in (
        ("movimiento", id_movimiento, id_documento),
        ("documento", id_documento, id_movimiento),
    ):
        conexion.execute(
            "UPDATE pendientes SET estado_revision = 'conciliado_manual', resuelto_con = ?, "
            "id_conciliacion_manual = ?, comentario_revision = ?, fecha_revision = ? "
            "WHERE id_ejecucion = ? AND lado = ? AND id_item = ?",
            (contraparte, id_conciliacion, comentario, ahora, id_ejecucion, lado, id_item),
        )

    conexion.commit()
    return id_conciliacion


def descartar_pendiente(
    conexion: sqlite3.Connection,
    id_ejecucion: int,
    id_item: str,
    motivo: str,
    lado: str = "movimiento",
) -> None:
    """Cierra un pendiente que NO corresponde a ningun documento.

    Es una resolucion legitima, no un error: una comision bancaria, un impuesto o
    un pago duplicado quedan revisados y explicados, sin inventar un match.
    """
    _pendiente(conexion, id_ejecucion, lado, id_item)
    if not motivo.strip():
        raise RevisionInvalidaError("Hay que indicar por que se descarta el pendiente")

    conexion.execute(
        "UPDATE pendientes SET estado_revision = 'descartado', comentario_revision = ?, "
        "fecha_revision = ? WHERE id_ejecucion = ? AND lado = ? AND id_item = ?",
        (motivo, datetime.now().isoformat(timespec="seconds"), id_ejecucion, lado, id_item),
    )
    conexion.commit()


def reabrir_pendiente(
    conexion: sqlite3.Connection, id_ejecucion: int, id_item: str, lado: str = "movimiento"
) -> None:
    """Deshace una revision: borra la conciliacion manual si la hubo y vuelve a pendiente."""
    fila = conexion.execute(
        "SELECT estado_revision, id_conciliacion_manual FROM pendientes "
        "WHERE id_ejecucion = ? AND lado = ? AND id_item = ?",
        (id_ejecucion, lado, id_item),
    ).fetchone()
    if fila is None:
        raise RevisionInvalidaError(
            f"{id_item} no figura como pendiente de la ejecucion {id_ejecucion}"
        )
    if fila[0] == "pendiente":
        return

    id_conciliacion = fila[1]
    if id_conciliacion:
        conexion.execute(
            "DELETE FROM conciliacion_items WHERE id_ejecucion = ? AND id_conciliacion = ?",
            (id_ejecucion, id_conciliacion),
        )
        conexion.execute(
            "DELETE FROM conciliaciones WHERE id_ejecucion = ? AND id_conciliacion = ?",
            (id_ejecucion, id_conciliacion),
        )

    # Reabre los dos lados: una conciliacion manual siempre involucra a ambos.
    conexion.execute(
        "UPDATE pendientes SET estado_revision = 'pendiente', resuelto_con = NULL, "
        "id_conciliacion_manual = NULL, comentario_revision = NULL, fecha_revision = NULL "
        "WHERE id_ejecucion = ? AND (id_conciliacion_manual = ? OR (lado = ? AND id_item = ?))",
        (id_ejecucion, id_conciliacion or "", lado, id_item),
    )
    conexion.commit()


def resumen_revision(conexion: sqlite3.Connection, id_ejecucion: int) -> dict:
    """Pendientes por lado y estado: {'movimiento': {'pendiente': 3, ...}, 'documento': {...}}."""
    resumen: dict[str, dict[str, int]] = {"movimiento": {}, "documento": {}}
    for lado, estado, cantidad in conexion.execute(
        "SELECT lado, estado_revision, COUNT(*) FROM pendientes WHERE id_ejecucion = ? "
        "GROUP BY lado, estado_revision",
        (id_ejecucion,),
    ):
        resumen.setdefault(lado, {})[estado] = cantidad
    return resumen
