"""Motor de conciliacion deterministico.

Esta es la pieza central del proyecto y NO usa IA: dados los mismos archivos de
entrada produce siempre el mismo resultado, y cada emparejamiento queda con la
estrategia que lo produjo y un puntaje de confianza que se puede auditar.

El motor aplica pasadas en orden de mayor a menor certeza. Cada pasada solo mira
los movimientos y documentos que las anteriores dejaron libres:

    1. exacto           RUT + monto + fecha identicos
    2. tolerancia_fecha RUT + monto identicos, fecha dentro de una ventana (merge_asof)
    3. comision         RUT + fecha en ventana, el abono llega por un poco menos
    4. fuzzy            sin RUT en la glosa: monto + fecha + similitud de nombre
    5. uno_a_n          un abono paga varias facturas (suma de subconjuntos acotada)
    6. n_a_1            varias cuotas pagan una factura

Lo que queda sin emparejar (o con confianza baja) se entrega como pendiente,
con los mejores candidatos calculados, para que lo resuelva ia.py con revision
humana. El LLM nunca decide si dos montos cuadran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import pandas as pd
from rapidfuzz import fuzz

# Confianza asignada por cada estrategia. El fuzzy la calcula a partir del score.
CONFIANZA = {
    "exacto": 1.00,
    "tolerancia_fecha": 0.95,
    "comision": 0.90,
    "uno_a_n": 0.88,
    "n_a_1": 0.88,
}

# Bajo este umbral el emparejamiento se propone, pero se marca para revision humana.
UMBRAL_AUTOMATICO = 0.85


@dataclass(frozen=True)
class ParametrosMatching:
    """Umbrales del motor. Estan afuera del codigo para poder calibrarlos y testearlos."""

    ventana_dias: int = 5
    """Cuantos dias puede demorar el banco en acreditar respecto de la fecha del DTE."""

    ventana_dias_cuotas: int = 20
    """Ventana mas amplia para pagos en cuotas: se reparten a lo largo de semanas, no de dias."""

    comision_maxima: float = 15_000
    """Diferencia absoluta maxima que se acepta como comision bancaria."""

    comision_maxima_pct: float = 0.01
    """Y ademas no puede superar este porcentaje del documento (evita 'calzar' por fuerza bruta)."""

    umbral_fuzzy: int = 85
    """Score minimo de similitud de nombre (rapidfuzz) para aceptar un match sin RUT."""

    umbral_fuzzy_candidato: int = 60
    """Bajo el umbral de aceptacion pero suficiente para ofrecerlo como candidato al LLM."""

    max_documentos_combinados: int = 4
    """Tope de facturas que puede cubrir un solo abono (acota el costo de la suma de subconjuntos)."""

    max_candidatos_combinacion: int = 12
    """Tope de candidatos que entran a la combinatoria (C(12,4) = 495: barato y suficiente)."""

    tolerancia_centavos: float = 1.0
    """Holgura para comparar sumas y evitar problemas de redondeo."""


@dataclass
class ResultadoConciliacion:
    """Salida del motor: lo conciliado y lo que queda pendiente."""

    conciliaciones: pd.DataFrame
    movimientos_pendientes: pd.DataFrame
    documentos_pendientes: pd.DataFrame


@dataclass
class _Emparejamiento:
    """Un match propuesto por alguna pasada del motor."""

    ids_movimientos: list[str]
    ids_documentos: list[str]
    estrategia: str
    confianza: float
    monto_banco: float
    monto_documentos: float
    detalle: str = ""
    score_nombre: float | None = None


@dataclass
class _Estado:
    """Lleva la cuenta de que movimientos y documentos siguen libres."""

    movimientos: pd.DataFrame
    documentos: pd.DataFrame
    usados_mov: set[str] = field(default_factory=set)
    usados_doc: set[str] = field(default_factory=set)
    emparejamientos: list[_Emparejamiento] = field(default_factory=list)

    def movimientos_libres(self) -> pd.DataFrame:
        return self.movimientos[~self.movimientos["id_movimiento"].isin(self.usados_mov)]

    def documentos_libres(self) -> pd.DataFrame:
        return self.documentos[~self.documentos["id_documento"].isin(self.usados_doc)]

    def registrar(self, emparejamiento: _Emparejamiento) -> bool:
        """Acepta el match solo si todas sus partes siguen libres (evita usar dos veces un DTE)."""
        if any(m in self.usados_mov for m in emparejamiento.ids_movimientos):
            return False
        if any(d in self.usados_doc for d in emparejamiento.ids_documentos):
            return False
        self.usados_mov.update(emparejamiento.ids_movimientos)
        self.usados_doc.update(emparejamiento.ids_documentos)
        self.emparejamientos.append(emparejamiento)
        return True


# Columnas que el motor espera de limpieza.py; sirven para no reventar con tablas vacias.
COLUMNAS_MOVIMIENTO = [
    "id_movimiento",
    "fecha",
    "descripcion",
    "glosa",
    "monto",
    "tipo",
    "monto_abs",
    "rut",
    "contraparte",
]
COLUMNAS_DOCUMENTO = [
    "id_documento",
    "folio",
    "tipo_dte",
    "fecha_emision",
    "rut",
    "razon_social",
    "contraparte",
    "monto_total",
]


def _asegurar_columnas(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Agrega las columnas que falten (caso tipico: un mes sin movimientos)."""
    faltantes = [c for c in columnas if c not in df.columns]
    if not faltantes:
        return df.copy()
    completo = df.copy()
    for columna in faltantes:
        completo[columna] = pd.Series(dtype="datetime64[ns]" if "fecha" in columna else "object")
    return completo


def _solo_abonos(movimientos: pd.DataFrame) -> pd.DataFrame:
    """Solo los abonos pueden pagar una venta; los cargos no se cruzan contra DTE."""
    return movimientos[movimientos["monto"] > 0].copy()


# --------------------------------------------------------------------------------------
# Pasada 1: match exacto
# --------------------------------------------------------------------------------------


def _pasada_exacta(estado: _Estado) -> None:
    """RUT + monto + fecha identicos: un merge directo de pandas lo resuelve."""
    movs = estado.movimientos_libres()
    movs = movs[movs["rut"] != ""]
    docs = estado.documentos_libres()
    if movs.empty or docs.empty:
        return

    pares = movs.merge(
        docs,
        left_on=["rut", "monto_abs", "fecha"],
        right_on=["rut", "monto_total", "fecha_emision"],
        suffixes=("_mov", "_doc"),
    ).sort_values(["fecha", "id_movimiento", "id_documento"])

    for fila in pares.itertuples():
        estado.registrar(
            _Emparejamiento(
                ids_movimientos=[fila.id_movimiento],
                ids_documentos=[fila.id_documento],
                estrategia="exacto",
                confianza=CONFIANZA["exacto"],
                monto_banco=fila.monto_abs,
                monto_documentos=fila.monto_total,
                detalle="RUT, monto y fecha identicos",
            )
        )


# --------------------------------------------------------------------------------------
# Pasada 2: tolerancia de fecha
# --------------------------------------------------------------------------------------


def _pasada_tolerancia_fecha(estado: _Estado, params: ParametrosMatching) -> None:
    """Mismo RUT y monto, pero el banco acredito unos dias despues.

    Se usa merge_asof, que empareja cada movimiento con el documento de fecha mas
    cercana dentro de la ventana; es la herramienta correcta de pandas para esto.
    """
    movs = estado.movimientos_libres()
    movs = movs[movs["rut"] != ""]
    docs = estado.documentos_libres()
    if movs.empty or docs.empty:
        return

    izquierda = movs.sort_values("fecha")
    derecha = docs.sort_values("fecha_emision")

    pares = pd.merge_asof(
        izquierda,
        derecha,
        left_on="fecha",
        right_on="fecha_emision",
        left_by=["rut", "monto_abs"],
        right_by=["rut", "monto_total"],
        direction="nearest",
        tolerance=pd.Timedelta(days=params.ventana_dias),
        suffixes=("_mov", "_doc"),
    ).dropna(subset=["id_documento"])

    pares["dias"] = (pares["fecha"] - pares["fecha_emision"]).dt.days.abs()
    for fila in pares.sort_values(["dias", "id_movimiento"]).itertuples():
        estado.registrar(
            _Emparejamiento(
                ids_movimientos=[fila.id_movimiento],
                ids_documentos=[fila.id_documento],
                estrategia="tolerancia_fecha",
                confianza=CONFIANZA["tolerancia_fecha"],
                monto_banco=fila.monto_abs,
                monto_documentos=fila.monto_total,
                detalle=f"Mismo RUT y monto, {int(fila.dias)} dia(s) de desfase",
            )
        )


# --------------------------------------------------------------------------------------
# Pasada 3: comision bancaria
# --------------------------------------------------------------------------------------


def _tolerancia_comision(monto_documento: float, params: ParametrosMatching) -> float:
    """La comision aceptada es la menor entre un tope fijo y un % del documento."""
    return min(params.comision_maxima, monto_documento * params.comision_maxima_pct)


def _pasada_comision(estado: _Estado, params: ParametrosMatching) -> None:
    """El abono llega por un poco menos que la factura: la diferencia es la comision.

    La diferencia no se 'esconde': queda registrada como partida conciliatoria.
    """
    movs = estado.movimientos_libres()
    movs = movs[movs["rut"] != ""]
    docs = estado.documentos_libres()
    if movs.empty or docs.empty:
        return

    pares = movs.merge(docs, on="rut", suffixes=("_mov", "_doc"))
    if pares.empty:
        return

    pares["dias"] = (pares["fecha"] - pares["fecha_emision"]).dt.days.abs()
    pares["diferencia"] = pares["monto_total"] - pares["monto_abs"]
    pares["tope"] = pares["monto_total"].map(lambda m: _tolerancia_comision(m, params))

    candidatos = pares[
        (pares["dias"] <= params.ventana_dias)
        & (pares["diferencia"] > 0)
        & (pares["diferencia"] <= pares["tope"])
    ]

    for fila in candidatos.sort_values(["diferencia", "dias", "id_movimiento"]).itertuples():
        estado.registrar(
            _Emparejamiento(
                ids_movimientos=[fila.id_movimiento],
                ids_documentos=[fila.id_documento],
                estrategia="comision",
                confianza=CONFIANZA["comision"],
                monto_banco=fila.monto_abs,
                monto_documentos=fila.monto_total,
                detalle=f"Diferencia de ${fila.diferencia:,.0f} atribuida a comision bancaria",
            )
        )


# --------------------------------------------------------------------------------------
# Pasada 4: fuzzy sobre el nombre
# --------------------------------------------------------------------------------------


def _candidatos_por_nombre(
    movimiento, documentos: pd.DataFrame, params: ParametrosMatching
) -> list[tuple[float, object]]:
    """Documentos con monto y fecha compatibles, ordenados por similitud de nombre."""
    if documentos.empty or not movimiento.contraparte:
        return []

    dias = (documentos["fecha_emision"] - movimiento.fecha).dt.days.abs()
    diferencia = documentos["monto_total"] - movimiento.monto_abs
    tope = documentos["monto_total"].map(lambda m: _tolerancia_comision(m, params))
    compatibles = documentos[
        (dias <= params.ventana_dias) & (diferencia >= 0) & (diferencia <= tope)
    ]

    puntuados = [
        (fuzz.token_set_ratio(movimiento.contraparte, doc.contraparte), doc)
        for doc in compatibles.itertuples()
    ]
    return sorted(puntuados, key=lambda par: par[0], reverse=True)


def _pasada_fuzzy(estado: _Estado, params: ParametrosMatching) -> None:
    """Cuando la glosa no trae RUT, se compara el nombre con rapidfuzz.

    El monto y la fecha siguen siendo condiciones duras: el fuzzy solo desempata
    entre documentos que ya calzan en plata. Nunca al reves.
    """
    for movimiento in estado.movimientos_libres().itertuples():
        candidatos = _candidatos_por_nombre(movimiento, estado.documentos_libres(), params)
        if not candidatos:
            continue

        score, doc = candidatos[0]
        if score < params.umbral_fuzzy:
            continue

        # Si el segundo candidato empata, no hay como decidir: se deja para revision.
        if len(candidatos) > 1 and candidatos[1][0] >= score:
            continue

        estado.registrar(
            _Emparejamiento(
                ids_movimientos=[movimiento.id_movimiento],
                ids_documentos=[doc.id_documento],
                estrategia="fuzzy",
                confianza=round(0.60 + 0.35 * (score / 100), 2),
                monto_banco=movimiento.monto_abs,
                monto_documentos=doc.monto_total,
                detalle=f"Similitud de nombre {score:.0f}/100 con monto y fecha compatibles",
                score_nombre=score,
            )
        )


# --------------------------------------------------------------------------------------
# Pasadas 5 y 6: N-a-N acotado
# --------------------------------------------------------------------------------------


def _subconjunto_que_suma(
    items: list, columna_monto: str, objetivo: float, params: ParametrosMatching
) -> list | None:
    """Suma de subconjuntos acotada: el subconjunto mas chico que suma el objetivo.

    Es un problema exponencial, asi que se acota por diseno: solo entran candidatos
    del mismo RUT dentro de la ventana de fechas, como maximo
    `max_candidatos_combinacion`, y en grupos de hasta `max_documentos_combinados`.
    """
    items = items[: params.max_candidatos_combinacion]
    for tamano in range(2, min(params.max_documentos_combinados, len(items)) + 1):
        for grupo in combinations(items, tamano):
            suma = sum(getattr(i, columna_monto) for i in grupo)
            if abs(suma - objetivo) <= params.tolerancia_centavos:
                return list(grupo)
    return None


def _pasada_uno_a_n(estado: _Estado, params: ParametrosMatching) -> None:
    """Un abono consolidado paga varias facturas del mismo cliente."""
    for movimiento in estado.movimientos_libres().itertuples():
        if not movimiento.rut:
            continue

        docs = estado.documentos_libres()
        candidatos = docs[
            (docs["rut"] == movimiento.rut)
            & ((docs["fecha_emision"] - movimiento.fecha).dt.days.abs() <= params.ventana_dias)
            & (docs["monto_total"] <= movimiento.monto_abs)
        ].sort_values("fecha_emision")

        grupo = _subconjunto_que_suma(
            list(candidatos.itertuples()), "monto_total", movimiento.monto_abs, params
        )
        if not grupo:
            continue

        estado.registrar(
            _Emparejamiento(
                ids_movimientos=[movimiento.id_movimiento],
                ids_documentos=[d.id_documento for d in grupo],
                estrategia="uno_a_n",
                confianza=CONFIANZA["uno_a_n"],
                monto_banco=movimiento.monto_abs,
                monto_documentos=sum(d.monto_total for d in grupo),
                detalle=f"Un abono cubre {len(grupo)} documentos del mismo RUT",
            )
        )


def _pasada_n_a_1(estado: _Estado, params: ParametrosMatching) -> None:
    """Varias cuotas pagan una misma factura."""
    for documento in estado.documentos_libres().itertuples():
        if not documento.rut:  # sin RUT no hay como acotar el grupo de cuotas
            continue

        movs = estado.movimientos_libres()
        candidatos = movs[
            (movs["rut"] == documento.rut)
            & (
                (movs["fecha"] - documento.fecha_emision).dt.days.abs()
                <= params.ventana_dias_cuotas
            )
            & (movs["monto_abs"] <= documento.monto_total)
        ].sort_values("fecha")

        grupo = _subconjunto_que_suma(
            list(candidatos.itertuples()), "monto_abs", documento.monto_total, params
        )
        if not grupo:
            continue

        estado.registrar(
            _Emparejamiento(
                ids_movimientos=[m.id_movimiento for m in grupo],
                ids_documentos=[documento.id_documento],
                estrategia="n_a_1",
                confianza=CONFIANZA["n_a_1"],
                monto_banco=sum(m.monto_abs for m in grupo),
                monto_documentos=documento.monto_total,
                detalle=f"{len(grupo)} abonos cubren un mismo documento",
            )
        )


# --------------------------------------------------------------------------------------
# Armado del resultado
# --------------------------------------------------------------------------------------

COLUMNAS_CONCILIACION = [
    "id_conciliacion",
    "estrategia",
    "confianza",
    "requiere_revision",
    "ids_movimientos",
    "ids_documentos",
    "n_movimientos",
    "n_documentos",
    "monto_banco",
    "monto_documentos",
    "diferencia",
    "detalle",
]


def _tabla_conciliaciones(estado: _Estado) -> pd.DataFrame:
    filas = []
    for i, emp in enumerate(estado.emparejamientos, start=1):
        diferencia = round(emp.monto_documentos - emp.monto_banco, 2)
        filas.append(
            {
                "id_conciliacion": f"CON-{i:04d}",
                "estrategia": emp.estrategia,
                "confianza": emp.confianza,
                "requiere_revision": emp.confianza < UMBRAL_AUTOMATICO,
                "ids_movimientos": "|".join(emp.ids_movimientos),
                "ids_documentos": "|".join(emp.ids_documentos),
                "n_movimientos": len(emp.ids_movimientos),
                "n_documentos": len(emp.ids_documentos),
                "monto_banco": emp.monto_banco,
                "monto_documentos": emp.monto_documentos,
                "diferencia": diferencia,
                "detalle": emp.detalle,
            }
        )
    return pd.DataFrame(filas, columns=COLUMNAS_CONCILIACION)


def _motivo_movimiento(movimiento, estado: _Estado, params: ParametrosMatching) -> str:
    """Clasifica por que un movimiento quedo pendiente (sin usar IA)."""
    if movimiento.monto < 0:
        return "cargo_sin_documento"

    conciliados = estado.movimientos[estado.movimientos["id_movimiento"].isin(estado.usados_mov)]
    gemelos = conciliados[
        (conciliados["rut"] == movimiento.rut) & (conciliados["monto_abs"] == movimiento.monto_abs)
    ]
    if movimiento.rut and not gemelos.empty:
        return "posible_pago_duplicado"

    if not movimiento.rut and not movimiento.contraparte:
        return "glosa_sin_contraparte"
    return "sin_documento_calzado"


def _tabla_movimientos_pendientes(estado: _Estado, params: ParametrosMatching) -> pd.DataFrame:
    filas = []
    for movimiento in estado.movimientos_libres().itertuples():
        candidatos = (
            _candidatos_por_nombre(movimiento, estado.documentos_libres(), params)
            if movimiento.monto > 0
            else []
        )
        sugeridos = [
            f"{doc.id_documento}:{score:.0f}"
            for score, doc in candidatos[:3]
            if score >= params.umbral_fuzzy_candidato
        ]
        filas.append(
            {
                "id_movimiento": movimiento.id_movimiento,
                "fecha": movimiento.fecha,
                "descripcion": movimiento.descripcion,
                "monto": movimiento.monto,
                "rut": movimiento.rut,
                "contraparte": movimiento.contraparte,
                "motivo": _motivo_movimiento(movimiento, estado, params),
                "candidatos": "|".join(sugeridos),
            }
        )
    return pd.DataFrame(
        filas,
        columns=[
            "id_movimiento",
            "fecha",
            "descripcion",
            "monto",
            "rut",
            "contraparte",
            "motivo",
            "candidatos",
        ],
    )


def _tabla_documentos_pendientes(estado: _Estado) -> pd.DataFrame:
    pendientes = estado.documentos_libres().copy()
    pendientes["motivo"] = "sin_pago_registrado"
    columnas = [
        "id_documento",
        "folio",
        "fecha_emision",
        "rut",
        "razon_social",
        "monto_total",
        "motivo",
    ]
    return pendientes.reindex(columns=columnas).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Punto de entrada
# --------------------------------------------------------------------------------------


def conciliar(
    movimientos: pd.DataFrame,
    documentos: pd.DataFrame,
    params: ParametrosMatching | None = None,
) -> ResultadoConciliacion:
    """Ejecuta el motor completo sobre datos ya limpios (ver limpieza.py).

    Devuelve las conciliaciones propuestas y los pendientes de ambos lados.
    """
    params = params or ParametrosMatching()
    movimientos = _asegurar_columnas(movimientos, COLUMNAS_MOVIMIENTO)
    documentos = _asegurar_columnas(documentos, COLUMNAS_DOCUMENTO)
    estado = _Estado(movimientos=_solo_abonos(movimientos), documentos=documentos)

    _pasada_exacta(estado)
    _pasada_tolerancia_fecha(estado, params)
    _pasada_comision(estado, params)
    _pasada_fuzzy(estado, params)
    _pasada_uno_a_n(estado, params)
    _pasada_n_a_1(estado, params)

    # Los cargos nunca se cruzan contra ventas, pero deben aparecer como pendientes.
    estado.movimientos = movimientos.copy()

    return ResultadoConciliacion(
        conciliaciones=_tabla_conciliaciones(estado),
        movimientos_pendientes=_tabla_movimientos_pendientes(estado, params),
        documentos_pendientes=_tabla_documentos_pendientes(estado),
    )
