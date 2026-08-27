"""Orquestador del pipeline completo: ingesta -> limpieza -> matching -> SQLite -> IA.

Existe para que el dashboard (app.py) no tenga logica de negocio: Streamlit
llama a `ejecutar()` igual que la linea de comandos.

    python -m src.pipeline
    python -m src.pipeline --cartola data/raw/cartola_banco.csv --ia
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ia import AnalizadorExcepciones, IANoConfiguradaError
from src.ingesta import leer_cartola, leer_libro_ventas
from src.limpieza import limpiar_cartola, limpiar_libro_ventas
from src.matching import ParametrosMatching, ResultadoConciliacion, conciliar
from src.persistencia import conectar, guardar_ejecucion, guardar_sugerencia_ia

RAW = Path("data") / "raw"
CARTOLA_POR_DEFECTO = RAW / "cartola_banco.csv"
VENTAS_POR_DEFECTO = RAW / "libro_ventas.csv"


@dataclass
class Ejecucion:
    """Todo lo que produjo una corrida del pipeline."""

    id_ejecucion: int | None
    movimientos: pd.DataFrame
    documentos: pd.DataFrame
    resultado: ResultadoConciliacion
    sugerencias_ia: pd.DataFrame
    errores_ia: list[str]


def ejecutar(
    ruta_cartola: str | Path = CARTOLA_POR_DEFECTO,
    ruta_ventas: str | Path = VENTAS_POR_DEFECTO,
    ruta_db: str | Path | None = None,
    params: ParametrosMatching | None = None,
    usar_ia: bool = False,
    limite_ia: int | None = None,
) -> Ejecucion:
    """Corre el pipeline de punta a punta.

    La IA es opcional y va al final: si falla o no esta configurada, la
    conciliacion deterministica ya esta completa y guardada.
    """
    movimientos = limpiar_cartola(leer_cartola(ruta_cartola))
    documentos = limpiar_libro_ventas(leer_libro_ventas(ruta_ventas))
    resultado = conciliar(movimientos, documentos, params)

    id_ejecucion = None
    conexion = None
    if ruta_db is not None:
        conexion = conectar(ruta_db)
        id_ejecucion = guardar_ejecucion(
            conexion, movimientos, documentos, resultado, str(ruta_cartola), str(ruta_ventas)
        )

    sugerencias = pd.DataFrame()
    errores: list[str] = []
    if usar_ia:
        analizador = AnalizadorExcepciones()
        try:
            sugerencias = analizador.analizar_pendientes(
                resultado.movimientos_pendientes, documentos, limite=limite_ia
            )
            errores = analizador.errores
        except IANoConfiguradaError as error:
            errores = [str(error)]

        if conexion is not None and not sugerencias.empty:
            for fila in sugerencias.itertuples():
                guardar_sugerencia_ia(
                    conexion,
                    id_ejecucion,
                    fila.id_movimiento,
                    fila.clasificacion,
                    fila.id_documento_sugerido,
                    fila.confianza,
                    fila.explicacion,
                )

    if conexion is not None:
        conexion.close()

    return Ejecucion(
        id_ejecucion=id_ejecucion,
        movimientos=movimientos,
        documentos=documentos,
        resultado=resultado,
        sugerencias_ia=sugerencias,
        errores_ia=errores,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Conciliacion bancaria de punta a punta")
    parser.add_argument("--cartola", default=str(CARTOLA_POR_DEFECTO))
    parser.add_argument("--ventas", default=str(VENTAS_POR_DEFECTO))
    parser.add_argument("--db", default=str(Path("data") / "conciliacion.db"))
    parser.add_argument("--ia", action="store_true", help="Analizar los pendientes con el LLM")
    parser.add_argument("--limite-ia", type=int, default=None)
    args = parser.parse_args()

    ejecucion = ejecutar(
        args.cartola, args.ventas, args.db, usar_ia=args.ia, limite_ia=args.limite_ia
    )
    resultado = ejecucion.resultado

    print(f"Ejecucion #{ejecucion.id_ejecucion} guardada en {args.db}\n")
    print(f"Movimientos leidos : {len(ejecucion.movimientos)}")
    print(f"Documentos leidos  : {len(ejecucion.documentos)}")
    print(f"Conciliaciones     : {len(resultado.conciliaciones)}")
    if not resultado.conciliaciones.empty:
        print(resultado.conciliaciones["estrategia"].value_counts().to_string())
    print(f"\nMovimientos pendientes: {len(resultado.movimientos_pendientes)}")
    if not resultado.movimientos_pendientes.empty:
        print(resultado.movimientos_pendientes["motivo"].value_counts().to_string())
    print(f"Documentos pendientes : {len(resultado.documentos_pendientes)}")

    if args.ia:
        print(f"\nSugerencias de IA: {len(ejecucion.sugerencias_ia)}")
        for error in ejecucion.errores_ia:
            print(f"  [IA] {error}")


if __name__ == "__main__":
    main()
