# Conciliador Bancario Inteligente para PyMEs — Brief del Proyecto

> Este archivo es el contexto base para Claude Code. Léelo completo antes de escribir código.
> Autor: Diego Lobos Ortiz — Ingeniería en Informática, mención Desarrollo de Software (UTUC, 3er año).
> Objetivo: proyecto de portafolio para conseguir trabajo como Software Engineer con foco en Data/AI Automation.

## 1. Qué estamos construyendo

Una aplicación que automatiza la **conciliación bancaria** de una PyME chilena: cruza la
**cartola bancaria** (lo que el banco dice que entró/salió) contra el **libro de ventas /
registro de documentos tributarios (DTE)** de la empresa (lo que la empresa registró), identifica
qué movimientos coinciden, detecta diferencias, y deja solo los casos ambiguos para revisión
humana asistida por IA.

**No es un proyecto de juguete tipo "to-do app".** Resuelve un problema real y medible: la
conciliación manual consume entre 4 y 20 horas al mes en una PyME chilena y es propensa a
errores que pueden generar inconsistencias en el F29 o la renta ante el SII.

## 2. Trasfondo del problema (resumen — contexto completo en el material de estudio)

- La conciliación bancaria es un control interno básico exigido implícitamente por la
  obligación de llevar "contabilidad fidedigna" (Código de Comercio) y es de los primeros
  documentos que pide el SII en una fiscalización.
- El 98%+ de las empresas en Chile son MiPymes, y la mayoría todavía concilia con Excel/manual;
  la brecha de digitalización pyme vs. gran empresa es significativa.
- Errores/casos típicos que el sistema debe manejar: pagos duplicados, un pago que cubre varias
  facturas (1-a-N), varios pagos que cubren una factura (N-a-1), diferencias de fecha entre banco
  y contabilidad, comisiones bancarias no registradas, glosas/descripciones ambiguas o mal escritas.
- Contexto tributario: DTE (documento tributario electrónico, el "respaldo" de cada venta) →
  alimenta el Registro de Compras y Ventas (RCV) → se declara en el F29 mensual ante el SII. El
  pago de esa venta entra al banco → la conciliación empareja el movimiento con el DTE.

## 3. Decisión de diseño más importante: determinismo primero, IA solo para excepciones

Esta es la regla que gobierna toda la arquitectura y la que Diego debe poder explicar en una
entrevista técnica:

- **El matching (exacto, aproximado, N-a-N, tolerancias de fecha/monto) se resuelve SIEMPRE con
  lógica de programación determinística (Pandas + rapidfuzz + reglas).** Debe ser reproducible,
  auditable y 100% consistente entre ejecuciones. Un LLM nunca decide si dos montos cuadran.
- **El LLM se usa SOLO para los casos que el motor determinístico no pudo resolver o marcó como
  ambiguos**: clasificar/explicar un movimiento sin glosa clara, sugerir el documento probable
  entre varios candidatos empatados, generar resúmenes en lenguaje natural, apoyar la detección
  de anomalías. Siempre con un campo de confianza y revisión humana antes de dar por conciliado
  un caso resuelto por IA.

## 4. Pipeline end-to-end

```
1. INGESTA       → leer CSV/Excel de cartola bancaria y de libro de ventas/DTE (pandas.read_csv/read_excel)
2. LIMPIEZA      → normalizar montos, fechas (datetime), RUT, glosas (mayúsculas, sin tildes/puntuación)
3. MATCHING      → (a) exact match determinístico (merge por rut+monto+fecha)
   DETERMINÍSTICO   (b) fuzzy match por glosa/nombre (rapidfuzz) con umbral de confianza
                     (c) N-a-N: suma de subconjuntos acotada por RUT + ventana de fechas
                     (d) tolerancia de fecha con merge_asof; tolerancia de monto por comisiones
4. EXCEPCIONES   → movimientos no conciliados o de baja confianza → LLM (Claude/OpenAI) con
   CON LLM          structured outputs / function calling → clasifica, explica, sugiere candidato
5. PERSISTENCIA  → SQLite: tablas movimientos, documentos, conciliaciones, pendientes
6. DASHBOARD     → Streamlit: KPIs, tabla conciliados/pendientes, filtros, descarga CSV
7. MÉTRICAS      → % conciliación automática, tiempo estimado ahorrado, precisión del matching
```

## 5. Stack tecnológico (y por qué)

| Componente | Elección | Por qué (para poder defenderlo en entrevista) |
|---|---|---|
| Lenguaje | Python 3.13 + venv | Estándar de la industria para datos/IA; ecosistema maduro |
| Manipulación de datos | Pandas | `merge`, `merge_asof`, `groupby` — el cruce y las tolerancias se resuelven ahí |
| Fuzzy matching | **rapidfuzz** (no thefuzz) | Licencia MIT (thefuzz es GPL), 5–100x más rápido (C++) |
| Base de datos | **SQLite** (no PostgreSQL) | Un solo proceso, sin concurrencia real; cero fricción de despliegue; migración a Postgres es un paso conocido si escalara |
| LLM | API directa de **Anthropic (Claude)** u **OpenAI**, con structured outputs/function calling | **Sin LangChain**: para una sola llamada con salida estructurada, LangChain añade complejidad sin beneficio |
| Modelo LLM sugerido | Claude Haiku 4.5 o GPT-4o mini | Baratos porque solo procesan las excepciones, no todo el volumen |
| Dashboard | **Streamlit** (no Power BI) | Todo en Python, reutiliza el pipeline, despliegue gratis en Streamlit Community Cloud, demuestra ingeniería end-to-end |
| Contenerización | Docker (ya lo usa Diego) | Buena práctica de portafolio; "funciona en cualquier máquina" |
| Testing | pytest | Cubrir la lógica de matching (el núcleo crítico) con `parametrize` y fixtures |

## 6. Estructura de carpetas objetivo

```
conciliador-bancario/
├── data/
│   ├── raw/                 # cartolas y ventas de ejemplo (CSV/Excel sintéticos)
│   └── processed/           # datos limpios
├── src/
│   ├── ingesta.py           # lectura CSV/Excel
│   ├── limpieza.py          # normalización de montos, fechas, RUT, glosas
│   ├── matching.py          # motor determinístico: exact, fuzzy, N-a-N, tolerancias
│   ├── ia.py                # cliente LLM (structured outputs) SOLO para excepciones
│   ├── persistencia.py      # SQLite: crear tablas, guardar/leer resultados
│   └── metricas.py          # % conciliado, tiempo ahorrado, precisión
├── app.py                   # dashboard Streamlit
├── tests/
│   ├── test_matching.py     # pruebas de la lógica de conciliación (pytest)
│   └── test_limpieza.py
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .env.example             # variables (API key) — nunca subir la real
├── CLAUDE.md                 # este archivo
└── README.md                 # problema, arquitectura, métricas, cómo correr
```

Principio: separación de responsabilidades. `matching.py` es independiente y 100% testeable sin
IA ni base de datos. `ia.py` es un módulo aislado que solo se invoca para excepciones. `app.py`
no contiene lógica de negocio, solo presenta.

## 7. Plan de desarrollo sugerido (fases)

1. **Fase 0 — Setup:** venv, requirements.txt, estructura de carpetas, repo Git, `.env.example`.
2. **Fase 1 — Datos sintéticos:** generar/preparar CSVs de ejemplo de cartola bancaria y libro
   de ventas con casos representativos (exactos, duplicados, 1-a-N, N-a-1, comisiones, desfases
   de fecha, glosas ambiguas).
3. **Fase 2 — Ingesta y limpieza:** `ingesta.py` + `limpieza.py`, con tests básicos.
4. **Fase 3 — Motor de matching determinístico:** exact match → fuzzy match → N-a-N → tolerancias.
   Esta es la parte más importante del proyecto; cubrir con pytest desde el principio.
5. **Fase 4 — Persistencia SQLite:** esquema de tablas, guardar resultados de matching.
6. **Fase 5 — Integración LLM:** cliente para excepciones, structured outputs, manejo de confianza.
7. **Fase 6 — Dashboard Streamlit:** KPIs, tabla interactiva, filtros, descarga de resultados.
8. **Fase 7 — Métricas, README y despliegue:** calcular métricas de éxito, escribir README con
   problema/arquitectura/decisiones, Dockerfile, desplegar en Streamlit Community Cloud.

## 8. Métricas de éxito (para el README y para medir el proyecto)

- % de transacciones conciliadas automáticamente (comparar contra el benchmark del sector: ~98%
  declarado por soluciones comerciales chilenas como referencia, siendo honestos sobre el
  resultado real logrado con los datos de prueba).
- Tiempo estimado ahorrado vs. proceso manual (base: 4–20 h/mes manual según volumen).
- Precisión del matching (matches correctos / matches propuestos, sobre un set etiquetado).
- Cobertura de tests de la lógica de matching.

## 9. Puntos que Diego debe poder explicar en una entrevista sobre este proyecto

1. Por qué el matching es determinístico y el LLM solo interviene en casos ambiguos con revisión
   humana (la decisión de diseño más importante del proyecto).
2. Trade-offs de tecnología: SQLite vs. PostgreSQL, API directa vs. LangChain, rapidfuzz vs.
   thefuzz, Streamlit vs. Power BI.
3. Cómo se resuelven los casos difíciles: 1-a-N/N-a-1 (suma de subconjuntos acotada), tolerancia
   de fechas (`merge_asof`), comisiones (diferencia residual como partida conciliatoria).
4. Limitaciones conocidas: datos sintéticos (no hay conexión real a bancos ni a la API del SII
   todavía), el LLM puede alucinar por lo que nunca decide aritmética, el matching N-a-N se acota
   con heurísticas por costo computacional, los umbrales de fuzzy matching requieren calibración.
5. Costos y escalabilidad: el LLM es barato porque solo procesa excepciones; si el proyecto
   escalara a multiusuario se migraría a PostgreSQL (ya está dockerizado, así que el camino a
   cloud está preparado).

## 10. Cómo trabajar en este proyecto (para Claude Code)

- Avanzar fase por fase (sección 7), no saltar directo al dashboard.
- Escribir tests (`pytest`) para `matching.py` a medida que se construye, no al final.
- Priorizar código simple y explicable por sobre abstracciones innecesarias — Diego debe poder
  defender cada decisión en una entrevista técnica.
- Mantener el LLM fuera de la lógica de matching determinístico, siempre.
- Antes de instalar una librería nueva, preguntar si se justifica frente al stack ya definido en
  la sección 5 (evitar dependencias innecesarias tipo LangChain para esta tarea).
