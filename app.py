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
from src.persistencia import (
    RevisionInvalidaError,
    conciliar_manualmente,
    conectar,
    descartar_pendiente,
    guardar_sugerencia_ia,
    leer_conciliaciones,
    leer_tabla,
    reabrir_pendiente,
    resumen_revision,
)
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


def leer_estado_guardado(id_ejecucion: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Lee de SQLite el estado vigente: incluye lo conciliado en revision humana."""
    conexion = conectar(DB)
    try:
        return (
            leer_conciliaciones(conexion, id_ejecucion),
            leer_tabla(conexion, "pendientes", id_ejecucion),
            resumen_revision(conexion, id_ejecucion),
        )
    finally:
        conexion.close()


def aplicar_revision(accion, *args, **kwargs) -> None:
    """Ejecuta una decision de revision y refresca la pantalla."""
    conexion = conectar(DB)
    try:
        accion(conexion, *args, **kwargs)
    except RevisionInvalidaError as error:
        st.error(str(error))
        return
    finally:
        conexion.close()
    st.rerun()


def guardar_sugerencias(id_ejecucion: int, sugerencias: pd.DataFrame) -> None:
    """Deja lo que propuso la IA junto al pendiente, para que aparezca en la revision."""
    if sugerencias.empty:
        return
    conexion = conectar(DB)
    try:
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
    finally:
        conexion.close()


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
    ruta_cartola = (
        guardar_subida(cartola_subida, "cartola") if cartola_subida else CARTOLA_POR_DEFECTO
    )
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
conciliaciones_db, pendientes_db, revision = leer_estado_guardado(ejecucion.id_ejecucion)
metricas = calcular(ejecucion.movimientos, ejecucion.documentos, resultado, revision)

# Estado de revision por item, para separar lo abierto de lo ya resuelto.
estados = pendientes_db.set_index("id_item") if not pendientes_db.empty else pd.DataFrame()


def estado_de(id_item: str, campo: str = "estado_revision", defecto: str = "pendiente"):
    if estados.empty or id_item not in estados.index:
        return defecto
    valor = estados.loc[id_item, campo]
    return defecto if pd.isna(valor) else valor


# ----------------------------------------------------------------------- KPIs

st.title("Conciliacion bancaria")
st.caption(
    f"Ejecucion #{ejecucion.id_ejecucion} · {metricas['total_movimientos']} movimientos · "
    f"{metricas['total_documentos']} documentos"
)

kpi = st.columns(5)
kpi[0].metric("Abonos cerrados", f"{metricas['pct_abonos_cerrados']}%",
              f"{metricas['movimientos_conciliados']} por el motor "
              f"+ {metricas['conciliados_manualmente']} revisados")
kpi[1].metric("Conciliaciones", len(conciliaciones_db),
              f"{metricas['n_conciliaciones']} del motor + "
              f"{metricas['conciliados_manualmente']} manuales")
kpi[2].metric("Sin revisar", metricas["pendientes_sin_revisar"],
              f"{metricas['pct_revisado']}% de los pendientes ya revisado", delta_color="off")
kpi[3].metric("Monto conciliado", pesos(metricas["monto_conciliado"]))
kpi[4].metric("Horas ahorradas (mes)", f"{metricas['horas_ahorradas']} h",
              f"vs {metricas['horas_manual']} h manual")

st.divider()


# ----------------------------------------------------------------------- tabs

tab_conciliado, tab_pendientes, tab_impagas, tab_metricas = st.tabs(
    [
        f"Conciliado ({len(conciliaciones_db)})",
        f"Pendientes ({len(resultado.movimientos_pendientes)})",
        f"Facturas impagas ({len(resultado.documentos_pendientes)})",
        "Metricas",
    ]
)

with tab_conciliado:
    # Se lee de la base, no de memoria: asi aparecen tambien las conciliaciones
    # creadas en revision humana.
    conciliaciones = conciliaciones_db.drop(columns=["id_ejecucion"], errors="ignore")
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
        pendientes = pendientes.assign(
            estado=pendientes["id_movimiento"].map(lambda i: estado_de(i)),
            resuelto_con=pendientes["id_movimiento"].map(
                lambda i: estado_de(i, "resuelto_con", "")
            ),
        )
        abiertos = pendientes[pendientes["estado"] == "pendiente"]
        resueltos = pendientes[pendientes["estado"] != "pendiente"]

        st.dataframe(
            abiertos.drop(columns=["estado", "resuelto_con"]),
            use_container_width=True,
            hide_index=True,
            column_config={"monto": st.column_config.NumberColumn("Monto", format="$%d")},
        )
        descargar(pendientes, "pendientes.csv", "Descargar pendientes (CSV)")

        # ------------------------------------------------------- revision humana
        st.subheader("Revision")
        st.caption(
            "El motor propone y la IA sugiere, pero un pendiente solo se cierra cuando una "
            "persona lo confirma. Lo confirmado queda como una conciliacion mas, con "
            "estrategia `revision_humana`."
        )

        if abiertos.empty:
            st.success("Todos los pendientes fueron revisados.")
        else:
            id_movimiento = st.selectbox(
                "Movimiento a revisar",
                abiertos["id_movimiento"],
                format_func=lambda i: (
                    f"{i} · {abiertos.set_index('id_movimiento').loc[i, 'descripcion'][:45]} · "
                    f"{pesos(abiertos.set_index('id_movimiento').loc[i, 'monto'])}"
                ),
            )
            elegido = abiertos.set_index("id_movimiento").loc[id_movimiento]
            st.caption(f"Motivo detectado por el motor: `{elegido['motivo']}`")

            sugerencia_ia = estado_de(id_movimiento, "sugerencia_ia", "")
            if sugerencia_ia:
                st.info(
                    f"La IA sugiere **{sugerencia_ia}** "
                    f"(confianza {estado_de(id_movimiento, 'confianza_ia', 0):.0%}): "
                    f"{estado_de(id_movimiento, 'explicacion_ia', '')}"
                )

            # Solo se ofrecen documentos que sigan libres: misma invariante que el motor.
            documentos_libres = [
                d
                for d in resultado.documentos_pendientes["id_documento"]
                if estado_de(d) == "pendiente"
            ]
            candidatos = [c.split(":")[0] for c in str(elegido["candidatos"] or "").split("|") if c]
            opciones = [d for d in candidatos if d in documentos_libres] + [
                d for d in documentos_libres if d not in candidatos
            ]

            resumen_documentos = resultado.documentos_pendientes.set_index("id_documento")
            columna_doc, columna_comentario = st.columns([1, 2])
            id_documento = columna_doc.selectbox(
                "Documento que corresponde",
                opciones,
                format_func=lambda d: (
                    f"{d} · {resumen_documentos.loc[d, 'razon_social'][:28]} · "
                    f"{pesos(resumen_documentos.loc[d, 'monto_total'])}"
                ),
                index=0 if opciones else None,
                placeholder="No hay documentos libres",
            )
            comentario = columna_comentario.text_input(
                "Comentario / motivo",
                placeholder="Ej: el cliente pago con la glosa mal escrita",
            )

            accion_izq, accion_der = st.columns(2)
            if accion_izq.button(
                "Conciliar con este documento", type="primary", use_container_width=True,
                disabled=not opciones,
            ):
                aplicar_revision(
                    conciliar_manualmente,
                    ejecucion.id_ejecucion,
                    id_movimiento,
                    id_documento,
                    comentario=comentario,
                )
            if accion_der.button(
                "Descartar (no corresponde a una venta)", use_container_width=True
            ):
                if not comentario.strip():
                    st.warning("Escribe el motivo antes de descartar el movimiento.")
                else:
                    aplicar_revision(
                        descartar_pendiente,
                        ejecucion.id_ejecucion,
                        id_movimiento,
                        comentario,
                    )

        if not resueltos.empty:
            with st.expander(f"Ya revisados ({len(resueltos)})"):
                for fila in resueltos.itertuples():
                    detalle = (
                        f"conciliado con **{fila.resuelto_con}**"
                        if fila.estado == "conciliado_manual"
                        else "descartado"
                    )
                    texto, boton = st.columns([4, 1])
                    texto.markdown(
                        f"`{fila.id_movimiento}` · {fila.descripcion[:50]} · "
                        f"{pesos(fila.monto)} — {detalle}"
                    )
                    if boton.button("Reabrir", key=f"reabrir_{fila.id_movimiento}"):
                        aplicar_revision(
                            reabrir_pendiente, ejecucion.id_ejecucion, fila.id_movimiento
                        )

        st.subheader("Analisis con IA")
        st.caption(
            "El LLM solo interviene aqui: clasifica el movimiento y, si hay candidatos, "
            "sugiere uno. Nunca decide montos y su respuesta siempre queda para revision humana."
        )

        analizador = AnalizadorExcepciones()
        if not analizador.disponible():
            st.info("Configura ANTHROPIC_API_KEY en un archivo .env para habilitar el analisis.")
        elif st.button("Analizar pendientes con IA", disabled=abiertos.empty):
            try:
                with st.spinner("Consultando al modelo..."):
                    sugerencias = analizador.analizar_pendientes(abiertos, ejecucion.documentos)
                st.session_state.sugerencias = sugerencias
                guardar_sugerencias(ejecucion.id_ejecucion, sugerencias)
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
    if not impagas.empty:
        impagas = impagas[impagas["id_documento"].map(lambda d: estado_de(d) == "pendiente")]
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
