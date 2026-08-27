"""Analisis de excepciones con LLM. SOLO se invoca para lo que el motor no resolvio.

Reglas de diseno de este modulo (son la parte defendible del proyecto):

1. El LLM NO hace aritmetica ni decide si dos montos cuadran. Eso ya lo resolvio
   matching.py de forma deterministica.
2. El LLM solo puede elegir entre los candidatos que el motor le entrega, o decir
   que ninguno corresponde. Si devuelve un id que no estaba en la lista, se
   descarta por codigo (guardrail): una alucinacion no puede entrar al resultado.
3. Toda sugerencia viene con confianza y queda marcada para revision humana. Nada
   se da por conciliado por decision de la IA.
4. Se usa la API de Anthropic directamente con structured outputs (json_schema).
   No se usa LangChain: para una sola llamada con salida estructurada solo
   agregaria dependencias y capas.

Como solo se procesan las excepciones (unas pocas decenas al mes), el costo es
marginal y se puede usar un modelo barato.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import pandas as pd

MODELO_POR_DEFECTO = "claude-haiku-4-5"

CLASIFICACIONES = [
    "pago_de_cliente",
    "pago_duplicado",
    "comision_o_gasto_bancario",
    "impuesto",
    "remuneraciones",
    "transferencia_interna",
    "abono_sin_respaldo",
    "otro",
]

INSTRUCCIONES = """Eres un asistente contable que ayuda a conciliar la cartola bancaria de una PyME chilena.

Un motor deterministico ya cruzo todo lo que se podia cruzar por RUT, monto y fecha.
Lo que recibes son SOLO los casos que ese motor no logro resolver.

Tu trabajo es acotado:
1. Clasificar de que tipo de movimiento se trata.
2. Si entre los documentos candidatos hay uno que corresponde, indicar su id.
3. Explicar en una o dos frases, en espanol y en lenguaje simple, por que.

Reglas estrictas:
- NO calcules ni valides montos: si un candidato calza en plata, el motor ya lo verifico.
- SOLO puedes sugerir un id_documento que aparezca en la lista de candidatos.
  Si ninguno corresponde, deja id_documento_sugerido vacio.
- Si no estas seguro, baja la confianza. Es preferible dejarlo pendiente que equivocarse:
  un contador va a revisar tu respuesta antes de dar el caso por conciliado.
- Un cargo (monto negativo) nunca corresponde a una venta."""

ESQUEMA_RESPUESTA = {
    "type": "object",
    "properties": {
        "clasificacion": {"type": "string", "enum": CLASIFICACIONES},
        "id_documento_sugerido": {
            "type": "string",
            "description": "Id de la lista de candidatos, o cadena vacia si ninguno corresponde",
        },
        "confianza": {"type": "number", "minimum": 0, "maximum": 1},
        "explicacion": {"type": "string"},
    },
    "required": ["clasificacion", "id_documento_sugerido", "confianza", "explicacion"],
    "additionalProperties": False,
}


class IANoConfiguradaError(RuntimeError):
    """No hay API key: el analisis de excepciones con LLM no esta disponible."""


@dataclass
class AnalisisPendiente:
    """Lo que el LLM propone para un pendiente. Siempre requiere revision humana."""

    id_movimiento: str
    clasificacion: str
    id_documento_sugerido: str
    confianza: float
    explicacion: str
    origen: str = "ia"
    requiere_revision: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalizadorExcepciones:
    """Cliente del LLM para los casos que el motor deterministico dejo pendientes."""

    modelo: str = MODELO_POR_DEFECTO
    max_tokens: int = 1024
    api_key: str | None = None
    cliente: object | None = None  # inyectable en los tests
    _errores: list[str] = field(default_factory=list, repr=False)

    # ---------------------------------------------------------------- cliente

    def _obtener_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:  # dotenv es opcional
            pass
        return os.getenv("ANTHROPIC_API_KEY")

    def _obtener_cliente(self):
        if self.cliente is not None:
            return self.cliente

        api_key = self._obtener_api_key()
        if not api_key:
            raise IANoConfiguradaError(
                "Falta ANTHROPIC_API_KEY. Copia .env.example a .env y completa la key, "
                "o revisa los pendientes manualmente en el dashboard."
            )

        import anthropic

        self.cliente = anthropic.Anthropic(api_key=api_key)
        return self.cliente

    def disponible(self) -> bool:
        """Permite que el dashboard muestre u oculte el boton de analisis con IA."""
        return self.cliente is not None or bool(self._obtener_api_key())

    # ------------------------------------------------------------- el prompt

    @staticmethod
    def _describir_caso(pendiente: dict, candidatos: list[dict]) -> str:
        movimiento = {
            "id_movimiento": pendiente.get("id_movimiento", ""),
            "fecha": str(pendiente.get("fecha", ""))[:10],
            "glosa_del_banco": pendiente.get("descripcion", ""),
            "monto": pendiente.get("monto", 0),
            "rut_detectado": pendiente.get("rut", "") or "(la glosa no trae RUT)",
            "motivo_por_el_que_quedo_pendiente": pendiente.get("motivo", ""),
        }
        return (
            "MOVIMIENTO BANCARIO SIN CONCILIAR:\n"
            f"{json.dumps(movimiento, ensure_ascii=False, indent=2)}\n\n"
            "DOCUMENTOS CANDIDATOS (el motor ya verifico que calzan en monto y fecha):\n"
            f"{json.dumps(candidatos, ensure_ascii=False, indent=2) if candidatos else 'Ninguno'}"
        )

    # ------------------------------------------------------------- guardrails

    @staticmethod
    def _validar(datos: dict, pendiente: dict, ids_validos: set[str]) -> AnalisisPendiente:
        """Filtra la respuesta del modelo antes de dejarla entrar al sistema."""
        sugerido = str(datos.get("id_documento_sugerido", "") or "").strip()
        explicacion = str(datos.get("explicacion", "")).strip()

        # Guardrail principal: un id inventado se descarta.
        if sugerido and sugerido not in ids_validos:
            explicacion = (
                f"{explicacion} [El modelo sugirio '{sugerido}', que no estaba entre los "
                "candidatos validos; la sugerencia fue descartada.]"
            ).strip()
            sugerido = ""

        clasificacion = str(datos.get("clasificacion", "otro"))
        if clasificacion not in CLASIFICACIONES:
            clasificacion = "otro"

        try:
            confianza = float(datos.get("confianza", 0))
        except (TypeError, ValueError):
            confianza = 0.0

        return AnalisisPendiente(
            id_movimiento=str(pendiente.get("id_movimiento", "")),
            clasificacion=clasificacion,
            id_documento_sugerido=sugerido,
            confianza=min(max(confianza, 0.0), 1.0),
            explicacion=explicacion,
        )

    # --------------------------------------------------------------- llamada

    def analizar_movimiento(self, pendiente: dict, candidatos: list[dict]) -> AnalisisPendiente:
        """Una llamada al LLM por movimiento pendiente, con salida estructurada."""
        cliente = self._obtener_cliente()

        respuesta = cliente.messages.create(
            model=self.modelo,
            max_tokens=self.max_tokens,
            system=INSTRUCCIONES,
            messages=[{"role": "user", "content": self._describir_caso(pendiente, candidatos)}],
            output_config={"format": {"type": "json_schema", "schema": ESQUEMA_RESPUESTA}},
        )

        texto = next((b.text for b in respuesta.content if b.type == "text"), "{}")
        datos = json.loads(texto)
        return self._validar(datos, pendiente, {c["id_documento"] for c in candidatos})

    # ------------------------------------------------------------ por lotes

    @staticmethod
    def _candidatos_de(pendiente: dict, documentos: pd.DataFrame) -> list[dict]:
        """Convierte la columna 'candidatos' ('DOC-1023:80|...') en datos legibles."""
        crudo = str(pendiente.get("candidatos", "") or "")
        if not crudo or documentos.empty:
            return []

        por_id = documentos.set_index("id_documento")
        candidatos = []
        for entrada in crudo.split("|"):
            id_documento, _, score = entrada.partition(":")
            if id_documento not in por_id.index:
                continue
            doc = por_id.loc[id_documento]
            candidatos.append(
                {
                    "id_documento": id_documento,
                    "fecha_emision": str(doc.get("fecha_emision", ""))[:10],
                    "razon_social": doc.get("razon_social", ""),
                    "rut": doc.get("rut", ""),
                    "monto_total": float(doc.get("monto_total", 0)),
                    "similitud_de_nombre": score,
                }
            )
        return candidatos

    def analizar_pendientes(
        self,
        movimientos_pendientes: pd.DataFrame,
        documentos: pd.DataFrame,
        limite: int | None = None,
    ) -> pd.DataFrame:
        """Analiza los pendientes uno por uno y devuelve una tabla de sugerencias.

        Si una llamada falla, el resto sigue: un error de red no puede dejar la
        conciliacion a medias.
        """
        self._errores = []
        if movimientos_pendientes.empty:
            return pd.DataFrame(columns=list(AnalisisPendiente.__annotations__))

        pendientes = movimientos_pendientes.head(limite) if limite else movimientos_pendientes
        analisis = []
        for pendiente in pendientes.to_dict("records"):
            try:
                resultado = self.analizar_movimiento(
                    pendiente, self._candidatos_de(pendiente, documentos)
                )
            except IANoConfiguradaError:
                raise
            except Exception as error:  # red, cuota, JSON malformado, etc.
                self._errores.append(f"{pendiente.get('id_movimiento', '?')}: {error}")
                continue
            analisis.append(resultado.as_dict())

        return pd.DataFrame(analisis, columns=list(AnalisisPendiente.__annotations__))

    @property
    def errores(self) -> list[str]:
        """Movimientos que no se pudieron analizar en la ultima corrida."""
        return list(self._errores)
