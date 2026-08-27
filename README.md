# Conciliador Bancario Inteligente para PyMEs

[![tests](https://github.com/Dlobosz/conciliacion-bancaria-pr/actions/workflows/tests.yml/badge.svg)](https://github.com/Dlobosz/conciliacion-bancaria-pr/actions/workflows/tests.yml)

Automatiza la **conciliación bancaria** de una PyME chilena: cruza la cartola del banco contra el
libro de ventas (DTE), identifica qué movimientos coinciden, detecta las diferencias y deja solo
los casos ambiguos para revisión humana asistida por IA.

Sobre el set de datos de prueba concilia **34 de los 36 pagos que tienen respaldo (94%)** con
**cero falsos positivos**, y clasifica los 10 movimientos restantes explicando por qué quedaron
pendientes.

```bash
python -m src.pipeline          # pipeline completo por consola
streamlit run app.py            # dashboard
```

---

## El problema

La conciliación bancaria consume entre **4 y 20 horas al mes** en una PyME chilena y se hace
mayoritariamente a mano en Excel. Es un control interno básico —deriva de la obligación de llevar
contabilidad fidedigna— y es de los primeros documentos que pide el SII en una fiscalización, así
que un error arrastra inconsistencias hasta el F29.

Los casos que hacen difícil el trabajo manual no son los pagos limpios, sino:

| Caso | Qué pasa |
|---|---|
| Desfase de fecha | El banco acredita 1 a 4 días después de emitida la factura |
| Comisión bancaria | Llega menos plata que la facturada; la diferencia no está registrada |
| 1-a-N | Un solo abono consolidado paga tres facturas |
| N-a-1 | Una factura grande se paga en cuotas a lo largo de semanas |
| Pago duplicado | El cliente paga dos veces; el segundo abono no calza con nada |
| Glosa ambigua | `TRANSF FERRETERIA INDUSTR` sin RUT: hay que adivinar de quién es |

---

## La decisión de diseño: determinismo primero, IA solo para excepciones

Esta es la regla que gobierna toda la arquitectura.

**El matching se resuelve siempre con lógica determinística** (Pandas + rapidfuzz + reglas). Dados
los mismos archivos produce siempre el mismo resultado, cada match queda con la estrategia y la
confianza que lo generaron, y todo es auditable. **Un LLM nunca decide si dos montos cuadran.**

**El LLM se invoca solo para lo que el motor no resolvió**: clasificar un movimiento sin glosa
clara, sugerir cuál de varios candidatos empatados corresponde, explicar el caso en lenguaje
natural. Y con tres restricciones duras implementadas en código, no en el prompt:

1. Solo puede sugerir un `id_documento` que el motor le pasó como candidato. Si devuelve otro, se
   descarta ([`src/ia.py`](src/ia.py) → `_validar`).
2. Toda sugerencia trae confianza acotada a `[0, 1]` y queda marcada `requiere_revision`.
3. Nada se da por conciliado por decisión de la IA.

Como solo procesa excepciones (10 casos de 44 movimientos), el costo del LLM es marginal.

---

## Arquitectura

```
data/raw/*.csv
     │
     ▼
 ingesta.py      lectura CSV/Excel, validación de columnas obligatorias
     │
     ▼
 limpieza.py     montos ($1.234.567 → float), fechas, RUT con dígito verificador,
     │           glosas sin tildes ni puntuación, extracción de RUT y contraparte
     ▼
 matching.py     6 pasadas en orden de certeza (ver abajo)  ← el núcleo, 98% cubierto por tests
     │
     ├──────────────► conciliaciones  ──► persistencia.py (SQLite)
     │
     └──► pendientes ──► ia.py (LLM, structured outputs) ──► sugerencias
                                     │
                                     ▼
                          revisión humana en el dashboard
                     (confirmar → conciliación, o descartar con motivo)
                                     │
                                     ▼
                        metricas.py + app.py (Streamlit)
```

### Las 6 pasadas del motor

Cada pasada solo mira lo que las anteriores dejaron libre, y ningún documento puede conciliarse
dos veces:

| # | Estrategia | Criterio | Confianza |
|---|---|---|---|
| 1 | `exacto` | RUT + monto + fecha idénticos (`merge`) | 1.00 |
| 2 | `tolerancia_fecha` | RUT + monto, fecha dentro de la ventana (`merge_asof` nearest) | 0.95 |
| 3 | `comision` | RUT + fecha, el abono llega por un poco menos; la diferencia queda registrada | 0.90 |
| 4 | `fuzzy` | Sin RUT en la glosa: monto y fecha compatibles + similitud de nombre (rapidfuzz) | 0.60–0.95 |
| 5 | `uno_a_n` | Suma de subconjuntos acotada por RUT y ventana de fechas | 0.88 |
| 6 | `n_a_1` | Varias cuotas suman una factura (ventana ampliada a 20 días) | 0.88 |

Las conciliaciones bajo `UMBRAL_AUTOMATICO` (0.90) se proponen pero se marcan
`requiere_revision`. El umbral está calibrado para que la marca signifique algo: por encima
quedan las pasadas que cruzan por identidad (RUT + monto) y las comisiones; por debajo caen las
dos familias que **infieren** —los N-a-N, porque una suma de subconjuntos puede cuadrar por
casualidad, y los fuzzy con score justo en el límite.

En las pasadas 4 a 6 **el monto es siempre condición dura**: el nombre solo desempata entre
documentos que ya calzan en plata, nunca al revés. Si dos candidatos empatan en similitud, el
motor no adivina: manda el caso a revisión.

La suma de subconjuntos es exponencial, así que se acota por diseño: solo entran documentos del
mismo RUT dentro de la ventana de fechas, máximo 12 candidatos, en grupos de hasta 4
(C(12,4) = 495 combinaciones, milisegundos).

### El ciclo se cierra con una persona

El motor propone y la IA sugiere, pero **un pendiente solo pasa a conciliado cuando alguien lo
confirma**. Desde el dashboard, cada movimiento pendiente tiene dos salidas:

- **Conciliar con un documento** → se crea una conciliación con estrategia `revision_humana` y
  confianza 1.0, indistinguible del resto para las métricas y las descargas. Se respeta la misma
  invariante que el motor: ni el movimiento ni el documento pueden estar ya usados en otra
  conciliación.
- **Descartar con motivo** → resolución legítima para una comisión bancaria, un impuesto o un pago
  duplicado: el caso queda cerrado y explicado, sin inventar un match.

Ambas decisiones se persisten (`estado_revision`, `resuelto_con`, `comentario_revision`,
`fecha_revision`) y son reversibles: *reabrir* borra la conciliación manual y devuelve los dos
lados a pendiente. Sobre el set de prueba, confirmar la glosa ambigua y descartar los 4 cargos
bancarios lleva la cobertura de 85% a 87,5% de abonos cerrados, con el 50% de los pendientes ya
revisados.

---

## Resultados sobre el set de prueba

44 movimientos bancarios contra 41 documentos, con los casos difíciles incluidos a propósito
(`data/raw/ground_truth.csv` trae el emparejamiento correcto para poder medir).

| Métrica | Resultado |
|---|---|
| **Precisión** (matches correctos / propuestos) | **100%** — 38/38, cero falsos positivos |
| **Cobertura** (correctos / esperados) | **95%** — 38 de 40 relaciones |
| Pagos con respaldo conciliados | 34 de 36 (94%) |
| Conciliaciones sin marca de revisión | 26 de 31 (84%) |
| Cobertura de tests de `matching.py` | 99% (97% del paquete `src/`, 133 tests) |
| Tiempo estimado ahorrado | ~87% del trabajo manual del período |

Desglose de las 31 conciliaciones: 14 exactas, 5 por tolerancia de fecha, 5 fuzzy, 3 de 1-a-N,
2 de N-a-1 y 2 con comisión bancaria.

Los 10 pendientes son correctos, no fallas: 4 cargos que nunca deberían calzar con una venta
(comisiones, impuestos, remuneraciones, arriendo), 2 pagos duplicados detectados como tales,
2 abonos sin respaldo documental y **2 glosas ambiguas** —`TRANSF FERRETERIA INDUSTR` y
`TRANSF C D`— que son exactamente los casos para los que existe el LLM.

**Sobre el 98% que declaran las soluciones comerciales:** ese número se logra con integración
directa a los bancos y al SII. Acá los datos son sintéticos y el 94% se alcanza sin ninguna
integración, priorizando precisión sobre cobertura: en conciliación un match incorrecto ensucia
la contabilidad, mientras que uno que falta solo queda pendiente de revisión.

---

## Stack y trade-offs

| Componente | Elección | Por qué |
|---|---|---|
| Datos | **Pandas** | `merge`, `merge_asof` y `groupby` resuelven el cruce y las tolerancias |
| Fuzzy matching | **rapidfuzz**, no thefuzz | Licencia MIT (thefuzz es GPL) e implementación en C++ |
| Base de datos | **SQLite**, no PostgreSQL | Un proceso, sin concurrencia real, cero fricción de despliegue. Migrar a Postgres es un cambio conocido si escalara |
| LLM | **API de Anthropic directa**, sin LangChain | Es una sola llamada con salida estructurada: LangChain agregaría capas sin beneficio |
| Modelo | **Claude Haiku 4.5** | Solo procesa excepciones, así que el modelo barato alcanza |
| Salida estructurada | `output_config` con `json_schema` | Garantiza el formato; los guardrails se validan igual en código |
| Dashboard | **Streamlit**, no Power BI | Todo en Python, reutiliza el pipeline y despliega gratis |
| Tests | **pytest** | `parametrize` + fixtures sobre la lógica crítica |

---

## Estructura

```
├── app.py                  dashboard Streamlit (sin lógica de negocio)
├── src/
│   ├── generar_datos.py    generador de datos sintéticos + ground truth
│   ├── ingesta.py          lectura y validación de CSV/Excel
│   ├── limpieza.py         normalización de montos, fechas, RUT y glosas
│   ├── matching.py         motor determinístico (el núcleo)
│   ├── ia.py               cliente LLM, solo para excepciones
│   ├── persistencia.py     SQLite (6 tablas) + ciclo de revisión humana
│   ├── metricas.py         KPIs, precisión y cobertura
│   └── pipeline.py         orquestador end-to-end + CLI
├── tests/                  133 tests
└── data/raw/               cartola, libro de ventas y ground truth sintéticos
```

`matching.py` es independiente y 100% testeable sin IA ni base de datos. `ia.py` está aislado y
solo se invoca para pendientes. `app.py` solo presenta.

---

## Cómo correr

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate en Linux/Mac)
pip install -r requirements-dev.txt   # requirements.txt solo para ejecutar

python -m src.generar_datos      # regenerar los datos sintéticos (opcional, ya vienen)
python -m src.pipeline           # conciliación completa por consola
python -m src.pipeline --ia      # incluyendo el análisis de pendientes con LLM
streamlit run app.py             # dashboard en http://localhost:8501

pytest                           # 133 tests
pytest --cov                     # con reporte de cobertura
ruff check src tests app.py      # linter
```

Para habilitar la IA, copiar `.env.example` a `.env` y completar `ANTHROPIC_API_KEY`. Sin la key
el pipeline funciona igual: la conciliación determinística está completa y los pendientes quedan
clasificados por el motor.

### Docker

```bash
docker build -t conciliador .
docker run -p 8501:8501 --env-file .env conciliador
```

La imagen pesa ~850 MB (pandas + pyarrow + streamlit) y trae el `HEALTHCHECK` apuntando a
`/_stcore/health`. Solo instala `requirements.txt`: las herramientas de test no se hornean en
la imagen.

---

## Limitaciones conocidas

- **Datos sintéticos.** No hay conexión real a bancos ni a la API del SII; la ingesta parte de
  CSV/Excel exportados a mano.
- **El LLM puede alucinar**, por eso nunca decide aritmética y su sugerencia se valida contra la
  lista de candidatos antes de entrar al sistema.
- **El matching N-a-N se acota con heurísticas** (mismo RUT, ventana de fechas, máximo 4
  documentos) por costo computacional: un pago consolidado de 6 facturas no lo resuelve.
- **Los umbrales requieren calibración** por empresa. El umbral fuzzy de 85 y la ventana de 5 días
  funcionan con estos datos; son ajustables desde el dashboard y desde `ParametrosMatching`.
- **Un solo período a la vez.** Un pago de una factura del mes anterior queda fuera de la ventana.
- **La imagen Docker corre como root.** Suficiente para una demo local; un despliegue real
  debería usar un usuario sin privilegios.
- **Sin autenticación ni multiusuario.** Es una herramienta de escritorio; escalar a multiusuario
  implica migrar a PostgreSQL (ya está dockerizado, así que el camino está preparado).

---

## Integración continua

Cada push a `main` y cada pull request dispara [el workflow de GitHub Actions](.github/workflows/tests.yml),
que corre cuatro pasos: linter (`ruff`), los 133 tests, un umbral mínimo de cobertura del 90% —hoy
está en 97%— y una verificación de que el generador de datos sintéticos sigue produciendo
exactamente los mismos archivos. Ese último paso protege las métricas del README: si el generador
cambia, las cifras medidas dejarían de corresponder a los datos versionados.

---

Proyecto de portafolio de **Diego Lobos Ortiz** — Ingeniería en Informática, mención Desarrollo de
Software.
