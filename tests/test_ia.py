"""Tests del modulo de IA.

No se llama a la API real: se inyecta un cliente falso. Lo que se prueba es el
contrato del modulo, sobre todo los guardrails que impiden que una alucinacion
del modelo entre al resultado.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.ia import AnalizadorExcepciones, IANoConfiguradaError


class ClienteFalso:
    """Imita anthropic.Anthropic devolviendo siempre la misma respuesta."""

    def __init__(self, respuesta: dict | Exception):
        self.respuesta = respuesta
        self.llamadas: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.llamadas.append(kwargs)
        if isinstance(self.respuesta, Exception):
            raise self.respuesta
        texto = json.dumps(self.respuesta, ensure_ascii=False)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=texto)])


PENDIENTE = {
    "id_movimiento": "MOV-0023",
    "fecha": pd.Timestamp("2025-06-14"),
    "descripcion": "TRANSF FERRETERIA INDUSTR",
    "monto": 2_032_800.0,
    "rut": "",
    "motivo": "sin_documento_calzado",
    "candidatos": "DOC-1023:80",
}

DOCUMENTOS = pd.DataFrame(
    [
        {
            "id_documento": "DOC-1023",
            "fecha_emision": pd.Timestamp("2025-06-14"),
            "rut": "76901234-1",
            "razon_social": "FERRETERIA INDUSTRIAL MAIPO LTDA",
            "monto_total": 2_032_800.0,
        }
    ]
)


def analizador(respuesta) -> tuple[AnalizadorExcepciones, ClienteFalso]:
    cliente = ClienteFalso(respuesta)
    return AnalizadorExcepciones(cliente=cliente), cliente


def test_respuesta_valida_se_mapea():
    ia, _ = analizador(
        {
            "clasificacion": "pago_de_cliente",
            "id_documento_sugerido": "DOC-1023",
            "confianza": 0.82,
            "explicacion": "La glosa truncada coincide con Ferreteria Industrial Maipo.",
        }
    )
    candidatos = ia._candidatos_de(PENDIENTE, DOCUMENTOS)
    analisis = ia.analizar_movimiento(PENDIENTE, candidatos)

    assert analisis.id_movimiento == "MOV-0023"
    assert analisis.id_documento_sugerido == "DOC-1023"
    assert analisis.confianza == 0.82
    assert analisis.origen == "ia"
    assert analisis.requiere_revision  # nunca se da por conciliado solo


def test_guardrail_descarta_un_documento_inventado():
    """Si el modelo devuelve un id que no estaba entre los candidatos, se ignora."""
    ia, _ = analizador(
        {
            "clasificacion": "pago_de_cliente",
            "id_documento_sugerido": "DOC-9999",
            "confianza": 0.95,
            "explicacion": "Corresponde a esa factura.",
        }
    )
    analisis = ia.analizar_movimiento(PENDIENTE, ia._candidatos_de(PENDIENTE, DOCUMENTOS))

    assert analisis.id_documento_sugerido == ""
    assert "descartada" in analisis.explicacion


def test_clasificacion_desconocida_cae_en_otro():
    ia, _ = analizador(
        {
            "clasificacion": "inventada_por_el_modelo",
            "id_documento_sugerido": "",
            "confianza": 0.5,
            "explicacion": "No se sabe.",
        }
    )
    analisis = ia.analizar_movimiento(PENDIENTE, [])
    assert analisis.clasificacion == "otro"


@pytest.mark.parametrize("cruda, esperada", [(1.7, 1.0), (-0.4, 0.0), ("alta", 0.0)])
def test_confianza_queda_acotada(cruda, esperada):
    ia, _ = analizador(
        {
            "clasificacion": "otro",
            "id_documento_sugerido": "",
            "confianza": cruda,
            "explicacion": "",
        }
    )
    assert ia.analizar_movimiento(PENDIENTE, []).confianza == esperada


def test_el_prompt_incluye_el_caso_y_los_candidatos():
    ia, cliente = analizador(
        {
            "clasificacion": "pago_de_cliente",
            "id_documento_sugerido": "DOC-1023",
            "confianza": 0.8,
            "explicacion": "ok",
        }
    )
    ia.analizar_movimiento(PENDIENTE, ia._candidatos_de(PENDIENTE, DOCUMENTOS))

    llamada = cliente.llamadas[0]
    contenido = llamada["messages"][0]["content"]
    assert "MOV-0023" in contenido
    assert "DOC-1023" in contenido
    assert "sin_documento_calzado" in contenido
    assert llamada["output_config"]["format"]["type"] == "json_schema"
    assert "NO calcules ni valides montos" in llamada["system"]


def test_candidatos_ignora_ids_que_no_existen():
    ia, _ = analizador({})
    pendiente = {**PENDIENTE, "candidatos": "DOC-1023:80|DOC-FANTASMA:70"}
    candidatos = ia._candidatos_de(pendiente, DOCUMENTOS)

    assert [c["id_documento"] for c in candidatos] == ["DOC-1023"]
    assert candidatos[0]["monto_total"] == 2_032_800.0


def test_sin_candidatos_no_falla():
    ia, _ = analizador({})
    assert ia._candidatos_de({**PENDIENTE, "candidatos": ""}, DOCUMENTOS) == []


def test_analizar_pendientes_devuelve_tabla():
    ia, _ = analizador(
        {
            "clasificacion": "pago_de_cliente",
            "id_documento_sugerido": "DOC-1023",
            "confianza": 0.8,
            "explicacion": "ok",
        }
    )
    tabla = ia.analizar_pendientes(pd.DataFrame([PENDIENTE]), DOCUMENTOS)

    assert len(tabla) == 1
    assert tabla.iloc[0]["id_documento_sugerido"] == "DOC-1023"


def test_un_error_de_red_no_bota_la_corrida():
    ia, _ = analizador(RuntimeError("connection reset"))
    tabla = ia.analizar_pendientes(pd.DataFrame([PENDIENTE]), DOCUMENTOS)

    assert tabla.empty
    assert len(ia.errores) == 1
    assert "MOV-0023" in ia.errores[0]


def test_pendientes_vacios_devuelve_tabla_vacia():
    ia, _ = analizador({})
    assert ia.analizar_pendientes(pd.DataFrame(), DOCUMENTOS).empty


def test_sin_api_key_falla_con_mensaje_util(monkeypatch):
    monkeypatch.setattr(AnalizadorExcepciones, "_obtener_api_key", lambda self: None)
    ia = AnalizadorExcepciones()

    assert not ia.disponible()
    with pytest.raises(IANoConfiguradaError, match="ANTHROPIC_API_KEY"):
        ia.analizar_movimiento(PENDIENTE, [])


def test_disponible_con_key_en_el_entorno(monkeypatch):
    monkeypatch.setattr(AnalizadorExcepciones, "_obtener_api_key", lambda self: "sk-ant-falsa")
    assert AnalizadorExcepciones().disponible()
