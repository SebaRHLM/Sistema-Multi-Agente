"""
Para salidas estructuradas.
"""

from typing import Literal
from pydantic import BaseModel, Field


class DecisionHerramienta(BaseModel):
    """Salida estructurada del LLM investigador."""

    herramienta_siguiente: Literal["calcular_map", "search_rag", "analizar"] = Field(
        description="Siguiente acción a ejecutar en el grafo."
    )
    justificacion_herramienta: str = Field(
        description="Razón clínica o lógica para usar esa herramienta o pasar al análisis."
    )


class RevisionJuez(BaseModel):
    """Salida estructurada del LLM juez."""

    estado_revision: Literal[
        "aprobado",
        "falta_evidencia",
        "falta_datos",
        "riesgo_alucinacion",
    ] = Field(description="Resultado de la revisión del análisis clínico preliminar.")
    observacion: str = Field(description="Explicación breve de la revisión.")
