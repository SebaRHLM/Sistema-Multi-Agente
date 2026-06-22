"""
Para salidas estructuradas.
"""

from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class DecisionHerramienta(BaseModel):
    """Salida estructurada del LLM investigador."""

    herramienta_siguiente: Literal["calcular_map", "search_rag", "clasificar_paciente_cardiovascular", "analizar"] = Field(
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

class DatosExtraidos_LLM_Extractor(BaseModel):
    sistolica: Optional[int] = Field(
        default=None,
        description="Presión arterial sistólica en mmHg."
    )
    diastolica: Optional[int] = Field(
        default=None,
        description="Presión arterial diastólica en mmHg."
    )
    trestbps: Optional[int] = Field(
        default=None,
        description="Presión arterial sistólica en reposo para el modelo ML cardiovascular."
    )

    age: Optional[int] = Field(
        default=None,
        description="Edad del paciente en años."
    )
    sex: Optional[int] = Field(
        default=None,
        description="Sexo codificado para modelo ML: 1 masculino, 0 femenino. None si no se indica."
    )

    sintomas: List[str] = Field(default_factory=list)
    antecedentes: List[str] = Field(default_factory=list)

    # Variables opcionales del modelo cardiovascular
    cp: Optional[float] = None
    chol: Optional[float] = None
    fbs: Optional[float] = None
    restecg: Optional[float] = None
    thalach: Optional[float] = None
    exang: Optional[float] = None
    oldpeak: Optional[float] = None
    slope: Optional[float] = None
    ca: Optional[float] = None
    thal: Optional[float] = None

    # Campos clínicos generales
    comorbilidades: List[str] = Field(default_factory=list)
    medicamentos: List[str] = Field(default_factory=list)
    examenes: List[str] = Field(default_factory=list)
    datos_faltantes: List[str] = Field(default_factory=list)