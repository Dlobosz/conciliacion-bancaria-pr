"""Generador de datos sinteticos: cartola bancaria + libro de ventas (DTE).

Produce tres archivos en data/raw/:
  - cartola_banco.csv   : lo que el banco dice que entro/salio
  - libro_ventas.csv    : los DTE que la empresa registro
  - ground_truth.csv    : el emparejamiento correcto, para medir precision del matching

Los datos cubren a proposito los casos dificiles que el motor de conciliacion
debe resolver: pagos exactos, desfases de fecha, comisiones bancarias, 1-a-N,
N-a-1, pagos duplicados, glosas ambiguas, movimientos sin documento y
documentos sin pago.

Ejecutar:  python -m src.generar_datos
"""

from __future__ import annotations

import random
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

SEED = 42
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
INICIO = date(2025, 6, 1)  # el periodo conciliado es junio 2025

CLIENTES = [
    ("76123456", "COMERCIAL LOS ALERCES SPA"),
    ("77234567", "DISTRIBUIDORA ANDINA LIMITADA"),
    ("96345678", "SUPERMERCADOS EL PARRON S.A."),
    ("77456789", "IMPORTADORA PACIFICO SUR SPA"),
    ("78567890", "CONSTRUCTORA VALLE VERDE LTDA"),
    ("76678901", "SERVICIOS INFORMATICOS NEXO SPA"),
    ("77789012", "TRANSPORTES CORDILLERA LIMITADA"),
    ("79890123", "AGRICOLA SANTA ELENA SPA"),
    ("76901234", "FERRETERIA INDUSTRIAL MAIPO LTDA"),
    ("77012345", "CLINICA DENTAL SONRISA SPA"),
    ("78345612", "PANADERIA DON LUCHO EIRL"),
    ("76456123", "LOGISTICA EXPRESS BIOBIO SPA"),
]


def digito_verificador(cuerpo: str) -> str:
    """Calcula el DV de un RUT chileno (modulo 11)."""
    suma, factor = 0, 2
    for digito in reversed(cuerpo):
        suma += int(digito) * factor
        factor = 2 if factor == 7 else factor + 1
    resto = 11 - (suma % 11)
    return {11: "0", 10: "K"}.get(resto, str(resto))


def formatear_rut(cuerpo: str) -> str:
    """76123456 -> 76.123.456-0 (formato tipico de un sistema contable)."""
    dv = digito_verificador(cuerpo)
    miles = f"{int(cuerpo):,}".replace(",", ".")
    return f"{miles}-{dv}"


def sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def ensuciar(razon_social: str, rng: random.Random) -> str:
    """Simula una glosa bancaria mal escrita o truncada."""
    palabras = razon_social.replace(".", "").split()
    estilo = rng.choice(["truncada", "sin_sociedad", "typo", "iniciales"])
    if estilo == "truncada":
        return " ".join(palabras[:2])[:18]
    if estilo == "sin_sociedad":
        sufijos = {"SPA", "S.A.", "SA", "LTDA", "LIMITADA", "EIRL"}
        return " ".join(p for p in palabras if p.upper() not in sufijos)
    if estilo == "typo":
        texto = list(" ".join(palabras))
        pos = rng.randrange(1, len(texto) - 1)
        texto[pos] = rng.choice("AEIOULRSTN")
        return "".join(texto)
    return " ".join(p[0] for p in palabras if len(p) > 2) + " " + palabras[-1]


class Generador:
    """Arma cartola, libro de ventas y ground truth caso por caso."""

    def __init__(self, seed: int = SEED) -> None:
        self.rng = random.Random(seed)
        self.documentos: list[dict] = []
        self.movimientos: list[dict] = []
        self.verdad: list[dict] = []
        self._folio = 1000
        self._n_mov = 0

    # ---------- helpers de construccion ----------

    def _fecha(self, dia: int) -> date:
        return INICIO + timedelta(days=dia)

    def _monto(self, minimo: int = 80_000, maximo: int = 4_500_000) -> int:
        """Monto total con IVA, redondeado a la centena (como un total real)."""
        return round(self.rng.randint(minimo, maximo), -2)

    def nuevo_documento(self, cliente: tuple[str, str], dia: int, total: int) -> dict:
        cuerpo, razon = cliente
        self._folio += 1
        neto = round(total / 1.19)
        doc = {
            "id_documento": f"DOC-{self._folio}",
            "tipo_dte": 33,  # factura electronica
            "folio": self._folio,
            "fecha_emision": self._fecha(dia).isoformat(),
            "rut_cliente": formatear_rut(cuerpo),
            "razon_social": razon,
            "monto_neto": neto,
            "iva": total - neto,
            "monto_total": total,
        }
        self.documentos.append(doc)
        return doc

    def nuevo_movimiento(self, dia: int, glosa: str, monto: int, tipo: str = "ABONO") -> dict:
        self._n_mov += 1
        mov = {
            "id_movimiento": f"MOV-{self._n_mov:04d}",
            "fecha": self._fecha(dia).isoformat(),
            "descripcion": sin_tildes(glosa).upper(),
            "tipo": tipo,
            "monto": monto if tipo == "ABONO" else -monto,
        }
        self.movimientos.append(mov)
        return mov

    def emparejar(self, mov: dict | None, doc: dict | None, caso: str) -> None:
        self.verdad.append(
            {
                "id_movimiento": mov["id_movimiento"] if mov else "",
                "id_documento": doc["id_documento"] if doc else "",
                "caso": caso,
            }
        )

    def glosa_pago(self, doc: dict, con_rut: bool = True) -> str:
        """Glosa de transferencia. Algunos bancos incluyen el RUT del pagador y otros no:
        cuando viene, el match exacto por rut+monto+fecha lo resuelve; cuando no,
        hay que caer al fuzzy matching sobre la razon social."""
        glosa = f"TRANSFERENCIA DE {doc['razon_social']}"
        return f"{glosa} RUT {doc['rut_cliente']}" if con_rut else glosa

    # ---------- casos ----------

    def caso_exacto(self, n: int = 12) -> None:
        """Mismo RUT, mismo monto, misma fecha: el merge directo debe resolverlo."""
        for i in range(n):
            cliente = CLIENTES[i % len(CLIENTES)]
            dia = self.rng.randrange(0, 28)
            doc = self.nuevo_documento(cliente, dia, self._monto())
            mov = self.nuevo_movimiento(dia, self.glosa_pago(doc), doc["monto_total"])
            self.emparejar(mov, doc, "exacto")

    def caso_desfase_fecha(self, n: int = 5) -> None:
        """El banco acredita 1 a 4 dias despues de emitido el documento."""
        for i in range(n):
            cliente = CLIENTES[(i + 3) % len(CLIENTES)]
            dia = self.rng.randrange(0, 24)
            doc = self.nuevo_documento(cliente, dia, self._monto())
            mov = self.nuevo_movimiento(
                dia + self.rng.randint(1, 4), self.glosa_pago(doc), doc["monto_total"]
            )
            self.emparejar(mov, doc, "desfase_fecha")

    def caso_comision(self, n: int = 4) -> None:
        """El banco descuenta una comision: el abono llega por menos que el documento."""
        for i in range(n):
            cliente = CLIENTES[(i + 5) % len(CLIENTES)]
            dia = self.rng.randrange(0, 26)
            doc = self.nuevo_documento(cliente, dia, self._monto())
            comision = self.rng.choice([2_500, 3_500, 5_000, 7_900])
            mov = self.nuevo_movimiento(
                dia + self.rng.randint(0, 2),
                self.glosa_pago(doc, con_rut=i % 2 == 0),
                doc["monto_total"] - comision,
            )
            self.emparejar(mov, doc, "comision_bancaria")

    def caso_glosa_ambigua(self, n: int = 5) -> None:
        """Monto y fecha calzan, pero la glosa viene truncada o mal escrita."""
        for i in range(n):
            cliente = CLIENTES[(i + 7) % len(CLIENTES)]
            dia = self.rng.randrange(0, 28)
            doc = self.nuevo_documento(cliente, dia, self._monto())
            mov = self.nuevo_movimiento(
                dia, f"TRANSF {ensuciar(doc['razon_social'], self.rng)}", doc["monto_total"]
            )
            self.emparejar(mov, doc, "glosa_ambigua")

    def caso_uno_a_n(self, grupos: int = 3) -> None:
        """Un solo abono paga varias facturas del mismo cliente."""
        for g in range(grupos):
            cliente = CLIENTES[(g + 2) % len(CLIENTES)]
            dia = self.rng.randrange(4, 22)
            docs = [
                self.nuevo_documento(
                    cliente, dia - self.rng.randint(0, 3), self._monto(90_000, 900_000)
                )
                for _ in range(self.rng.randint(2, 3))
            ]
            total = sum(d["monto_total"] for d in docs)
            mov = self.nuevo_movimiento(
                dia + 1, f"PAGO CONSOLIDADO {cliente[1]} RUT {docs[0]['rut_cliente']}", total
            )
            for doc in docs:
                self.emparejar(mov, doc, "uno_a_n")

    def caso_n_a_1(self, grupos: int = 2) -> None:
        """Una factura grande se paga en varias cuotas."""
        for g in range(grupos):
            cliente = CLIENTES[(g + 9) % len(CLIENTES)]
            dia = self.rng.randrange(0, 18)
            total = self._monto(1_500_000, 4_000_000)
            doc = self.nuevo_documento(cliente, dia, total)
            cuotas = self.rng.randint(2, 3)
            cortes = sorted(self.rng.sample(range(50_000, total - 50_000), cuotas - 1))
            montos = [b - a for a, b in zip([0] + cortes, cortes + [total])]
            for i, monto in enumerate(montos):
                mov = self.nuevo_movimiento(
                    dia + i * 3 + 1,
                    f"ABONO CUOTA {i + 1} {cliente[1]} RUT {doc['rut_cliente']}",
                    monto,
                )
                self.emparejar(mov, doc, "n_a_1")

    def caso_pago_duplicado(self, n: int = 2) -> None:
        """El cliente paga dos veces la misma factura: el segundo abono queda pendiente."""
        for i in range(n):
            cliente = CLIENTES[(i + 4) % len(CLIENTES)]
            dia = self.rng.randrange(0, 20)
            doc = self.nuevo_documento(cliente, dia, self._monto(200_000, 1_200_000))
            mov = self.nuevo_movimiento(dia, self.glosa_pago(doc), doc["monto_total"])
            self.emparejar(mov, doc, "exacto")
            duplicado = self.nuevo_movimiento(
                dia + self.rng.randint(1, 5), self.glosa_pago(doc), doc["monto_total"]
            )
            self.emparejar(duplicado, None, "pago_duplicado")

    def caso_sin_documento(self) -> None:
        """Movimientos bancarios que no corresponden a ninguna venta."""
        cargos = [
            ("COMISION MANTENCION CUENTA CORRIENTE", 9_900),
            ("IMPUESTO LEY TIMBRES Y ESTAMPILLAS", 12_450),
            ("PAGO REMUNERACIONES NOMINA", 3_280_000),
            ("CARGO ARRIENDO OFICINA INMOBILIARIA RAUCO", 850_000),
        ]
        for glosa, monto in cargos:
            mov = self.nuevo_movimiento(self.rng.randrange(0, 28), glosa, monto, tipo="CARGO")
            self.emparejar(mov, None, "sin_documento")

        abonos = [
            ("DEPOSITO EN EFECTIVO CAJA SUCURSAL", 145_000),
            ("TRANSFERENCIA RECIBIDA SIN GLOSA", 268_400),
        ]
        for glosa, monto in abonos:
            mov = self.nuevo_movimiento(self.rng.randrange(0, 28), glosa, monto)
            self.emparejar(mov, None, "sin_documento")

    def caso_sin_pago(self, n: int = 4) -> None:
        """Facturas emitidas que aun no han sido pagadas."""
        for i in range(n):
            cliente = CLIENTES[(i + 6) % len(CLIENTES)]
            doc = self.nuevo_documento(cliente, self.rng.randrange(18, 29), self._monto())
            self.emparejar(None, doc, "sin_pago")

    # ---------- salida ----------

    def construir(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.caso_exacto()
        self.caso_desfase_fecha()
        self.caso_comision()
        self.caso_glosa_ambigua()
        self.caso_uno_a_n()
        self.caso_n_a_1()
        self.caso_pago_duplicado()
        self.caso_sin_documento()
        self.caso_sin_pago()

        cartola = pd.DataFrame(self.movimientos).sort_values(["fecha", "id_movimiento"])
        cartola["saldo"] = 8_500_000 + cartola["monto"].cumsum()
        cartola = cartola[["fecha", "descripcion", "tipo", "monto", "saldo", "id_movimiento"]]

        ventas = pd.DataFrame(self.documentos).sort_values(["fecha_emision", "folio"])
        verdad = pd.DataFrame(self.verdad)
        return cartola, ventas, verdad


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    cartola, ventas, verdad = Generador().construir()

    cartola.to_csv(RAW / "cartola_banco.csv", index=False, encoding="utf-8")
    ventas.to_csv(RAW / "libro_ventas.csv", index=False, encoding="utf-8")
    verdad.to_csv(RAW / "ground_truth.csv", index=False, encoding="utf-8")

    print(f"cartola_banco.csv : {len(cartola)} movimientos")
    print(f"libro_ventas.csv  : {len(ventas)} documentos")
    print(f"ground_truth.csv  : {len(verdad)} relaciones esperadas")
    print("\nCasos generados:")
    print(verdad["caso"].value_counts().to_string())


if __name__ == "__main__":
    main()
