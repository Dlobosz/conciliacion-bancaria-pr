"""Dashboard de conciliacion bancaria (Streamlit).

Este archivo NO tiene logica de negocio: solo presenta lo que devuelve
src/pipeline.py. Toda decision de conciliacion vive en src/matching.py.

    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ia import AnalizadorExcepciones, IANoConfiguradaError
from src.ingesta import ArchivoInvalidoError
from src.matching import ParametrosMatching
from src.metricas import calcular, precision_contra_verdad
from src.pipeline import CARTOLA_POR_DEFECTO, VENTAS_POR_DEFECTO, ejecutar

RAW = Path("data") / "raw"
DB = Path("data") / "conciliacion.db"

st.set_page_config(page_title="Conciliador Bancario", page_icon="🏦", layout="wide")


def pesos(monto: float) -> str:
    """1234567 -> $1.234.567 (formato chileno)."""
    return f"${monto:,.0f}".replace(",", ".")


def guardar_subida(archivo, nombre: str) -> Path:
    """Deja un archivo subido en disco para que la ingesta lo lea como cualquier otro."""
    destino = Path(tempfile.gettempdir()) / f"conciliador_{nombre}{Path(archivo.name).suffix}"
    destino.write_bytes(archivo.getbuffer())
    return destino


def descargar(df: pd.DataFrame, nombre: str, etiqueta: str) -> None:
    st.download_button(
        etiqueta,
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name=nombre,
        mime="text/csv",
    )


# ----------------------------------------------------------------- barra lateral

st.sidebar.title("🏦 Conciliador Bancario")
st.sidebar.caption("Conciliacion determinista + IA solo para las excepciones")

cartola_subida = st.sidebar.file_uploader("Cartola bancaria (CSV/Excel)", type=["csv", "xlsx"])
ventas_subidas = st.sidebar.file_uploader("Libro de ventas / DTE (CSV/Excel)", type=["csv", "xlsx"])

st.sidebar.subheader("Parametros del motor")
ventana = st.sidebar.slider(
    "Ventana de fechas (dias)", 0, 15, 5, help="Cuanto puede demorar el banco en acreditar"
)
umbral_fuzzy = st.sidebar.slider(
    "Umbral de similitud de nombre", 60, 100, 85, help="Bajo este score el caso queda pendiente"
)
comision_maxima = st.sidebar.number_input(
    "Comision maxima aceptada ($)", min_value=0, max_value=100_000, value=15_000, step=500
)

conciliar_ahora = st.sidebar.button("Conciliar", type="primary", use_container_width=True)
st.sidebar.divider()
st.sidebar.caption(
    "Sin archivos cargados se usa el set de datos sinteticos de `data/raw/` "
    "(44 movimientos, 41 documentos)."
)


# ------------------------------------------------------------------ ejecucion

if conciliar_ahora or "ejecucion" not in st.session_state:
    ruta_cartola = guardar_subida(cartola_subida, "cartola") if cartola_subida else CARTOLA_POR_DEFECTO
    ruta_ventas = guardar_subida(ventas_subidas, "ventas") if ventas_subidas else VENTAS_POR_DEFECTO

    parametros = ParametrosMatching(
        ventana_dias=ventana, umbral_fuzzy=umbral_fuzzy, comision_maxima=comision_maxima
    )
    try:
        with st.spinner("Conciliando..."):
            st.session_state.ejecucion = ejecutar(
                ruta_cartola, ruta_ventas, ruta_db=DB, params=parametros
            )
        st.session_state.pop("sugerencias", None)
    except ArchivoInvalidoError as error:
        st.error(f"No se pudo leer el archivo: {error}")
        st.stop()

ejecucion = st.session_state.ejecucion
resultado = ejecucion.resultado
metricas = calcular(ejecucion.movimientos, ejecucion.documentos, resultado)


# ----------------------------------------------------------------------- KPIs

st.title("Conciliacion bancaria")
st.caption(
    f"Ejecucion #{ejecucion.id_ejecucion} · {metricas['total_movimientos']} movimientos · "
    f"{metricas['total_documentos']} documentos"
)

kpi = st.columns(5)
kpi[0].metric("Abonos conciliados", f"{metricas['pct_abonos_conciliados']}%",
              f"{metricas['movimientos_conciliados']} de {metricas['total_abonos']}")
kpi[1].metric("Conciliaciones", metricas["n_conciliaciones"],
              f"{metricas['pct_automatico']}% automaticas")
kpi[2].metric("Pendientes de revision", metricas["movimientos_pendientes"],
              f"{metricas['documentos_pendientes']} facturas impagas", delta_color="off")
kpi[3].metric("Monto conciliado", pesos(metricas["monto_conciliado"]))
kpi[4].metric("Horas ahorradas (mes)", f"{metricas['horas_ahorradas']} h",
              f"vs {metricas['horas_manual']} h manual")

st.divider()


# ----------------------------------------------------------------------- tabs

tab_conciliado, tab_pendientes, tab_impagas, tab_metricas = st.tabs(
    [
        f"Conciliado ({len(resultado.conciliaciones)})",
        f"Pendientes ({len(resultado.movimientos_pendientes)})",
        f"Facturas impagas ({len(resultado.documentos_pendientes)})",
        "Metricas",
    ]
)

with tab_conciliado:
    conciliaciones = resultado.conciliaciones
    if conciliaciones.empty:
        st.info("No se concilio ningun movimiento con los parametros actuales.")
    else:
        columna_izq, columna_der = st.columns([1, 2])
        estrategias = ["(todas)"] + sorted(conciliaciones["estrategia"].unique())
        estrategia = columna_izq.selectbox("Estrategia", estrategias)
        solo_revision = columna_der.checkbox("Solo las que requieren revision")

        vista = conciliaciones
        if estrategia != "(todas)":
            vista = vista[vista["estrategia"] == estrategia]
        if solo_revision:
            vista = vista[vista["requiere_revision"]]

        st.dataframe(
            vista,
            use_container_width=True,
            hide_index=True,
            column_config={
                "confianza": st.column_config.ProgressColumn(
                    "Confianza", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "monto_banco": st.column_config.NumberColumn("Monto banco", format="$%d"),
                "monto_documentos": st.column_config.NumberColumn("Monto documentos", format="$%d"),
                "diferencia": st.column_config.NumberColumn("Diferencia", format="$%d"),
            },
        )
        descargar(vista, "conciliaciones.csv", "Descargar conciliaciones (CSV)")

with tab_pendientes:
    pendientes = resultado.movimientos_pendientes
    if pendientes.empty:
        st.success("No quedaron movimientos pendientes.")
    else:
        st.dataframe(
            pendientes,
            use_container_width=True,
            hide_index=True,
            column_config={"monto": st.column_config.NumberColumn("Monto", format="$%d")},
        )
        descargar(pendientes, "pendientes.csv", "Descargar pendientes (CSV)")

        st.subheader("Analisis con IA")
        st.caption(
            "El LLM solo interviene aqui: clasifica el movimiento y, si hay candidatos, "
            "sugiere uno. Nunca decide montos y su respuesta siempre queda para revision humana."
        )

        analizador = AnalizadorExcepciones()
        if not analizador.disponible():
            st.info("Configura ANTHROPIC_API_KEY en un archivo .env para habilitar el analisis.")
        elif st.button("Analizar pendientes con IA"):
            try:
                with st.spinner("Consultando al modelo..."):
                    st.session_state.sugerencias = analizador.analizar_pendientes(
                        pendientes, ejecucion.documentos
                    )
                for error in analizador.errores:
                    st.warning(f"No se pudo analizar {error}")
            except IANoConfiguradaError as error:
                st.error(str(error))

        sugerencias = st.session_state.get("sugerencias")
        if sugerencias is not None and not sugerencias.empty:
            st.dataframe(
                sugerencias,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "confianza": st.column_config.ProgressColumn(
                        "Confianza", min_value=0.0, max_value=1.0, format="%.2f"
                    )
                },
            )
            descargar(sugerencias, "sugerencias_ia.csv", "Descargar sugerencias (CSV)")

with tab_impagas:
    impagas = resultado.documentos_pendientes
    if impagas.empty:
        st.success("Todas las facturas del periodo tienen pago registrado.")
    else:
        st.dataframe(
            impagas,
            use_container_width=True,
            hide_index=True,
            column_config={"monto_total": st.column_config.NumberColumn("Monto", format="$%d")},
        )
        st.metric("Total por cobrar", pesos(impagas["monto_total"].sum()))
        descargar(impagas, "facturas_impagas.csv", "Descargar facturas impagas (CSV)")

with tab_metricas:
    izquierda, derecha = st.columns(2)

    with izquierda:
        st.subheader("Como se concilio")
        por_estrategia = pd.Series(metricas["por_estrategia"], name="conciliaciones")
        if not por_estrategia.empty:
            st.bar_chart(por_estrategia)

    with derecha:
        st.subheader("Por que quedo pendiente")
        por_motivo = pd.Series(metricas["por_motivo_pendiente"], name="movimientos")
        if not por_motivo.empty:
            st.bar_chart(por_motivo)

    st.subheader("Diferencias detectadas")
    st.caption(
        "Diferencias absorbidas como comision bancaria. No se esconden: quedan "
        "registradas como partida conciliatoria."
    )
    st.metric("Total de diferencias", pesos(metricas["diferencias_por_comision"]))

    verdad = RAW / "ground_truth.csv"
    if verdad.exists() and not cartola_subida:
        st.subheader("Precision sobre el set etiquetado")
        st.caption(
            "Solo disponible con los datos sinteticos, que traen el emparejamiento correcto."
        )
        p = precision_contra_verdad(resultado, pd.read_csv(verdad, keep_default_na=False))
        columnas = st.columns(3)
        columnas[0].metric("Precision", f"{p['precision']:.0%}",
                           f"{len(p['falsos_positivos'])} falsos positivos")
        columnas[1].metric("Cobertura", f"{p['cobertura']:.0%}",
                           f"{len(p['no_encontrados'])} sin encontrar")
        columnas[2].metric("Matches correctos", f"{p['correctos']}/{p['esperados']}")
