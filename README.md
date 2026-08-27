# Conciliador Bancario Inteligente para PyMEs

Automatiza la conciliación bancaria de una PyME chilena: cruza la **cartola bancaria** contra el
**libro de ventas / registro de DTE**, identifica qué movimientos coinciden, detecta diferencias y
deja solo los casos ambiguos para revisión humana asistida por IA.

## Decisión de diseño central

**Determinismo primero, IA solo para excepciones.**

- El matching (exacto, fuzzy, N-a-N, tolerancias de fecha y monto) se resuelve **siempre** con
  lógica determinística (Pandas + rapidfuzz + reglas): reproducible, auditable y consistente
  entre ejecuciones. Un LLM nunca decide si dos montos cuadran.
- El LLM interviene **solo** en los casos que el motor determinístico no resolvió o marcó como
  ambiguos, siempre con un puntaje de confianza y revisión humana antes de dar por conciliado.

## Arquitectura

```
1. INGESTA      -> lectura de CSV/Excel (cartola bancaria y libro de ventas)
2. LIMPIEZA     -> normalización de montos, fechas, RUT y glosas
3. MATCHING     -> exacto -> fuzzy -> N-a-N -> tolerancias (determinístico)
4. EXCEPCIONES  -> LLM con structured outputs, solo para casos ambiguos
5. PERSISTENCIA -> SQLite
6. DASHBOARD    -> Streamlit (KPIs, tablas, filtros, descarga CSV)
7. MÉTRICAS     -> % conciliación automática, tiempo ahorrado, precisión
```

## Stack

Python · Pandas · rapidfuzz · SQLite · API directa de Anthropic (sin LangChain) · Streamlit ·
Docker · pytest

## Cómo correr

```bash
python -m venv venv
venv\Scripts\activate        # Windows  (source venv/bin/activate en Linux/Mac)
pip install -r requirements.txt

copy .env.example .env       # y completar la API key

pytest                       # tests
streamlit run app.py         # dashboard
```

## Estado

En desarrollo. Ver `CLAUDE.md` para el plan de fases completo.
