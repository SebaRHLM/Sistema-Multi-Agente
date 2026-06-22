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
    sex: Optional[float] = Field(
        default=None,
        description="Sexo del paciente. 1=male/masculino, 0=female/femenino. None si no aparece.",
    )

    sintomas: List[str] = Field(default_factory=list)
    antecedentes: List[str] = Field(default_factory=list)

    # Variables opcionales del modelo cardiovascular
    cp: Optional[float] = Field(
        default=None,
        description=(
            "Tipo de dolor torácico. Valores permitidos: "
            "0=typical angina, 1=atypical angina, "
            "2=non-anginal, 3=asymptomatic. "
            "Usar None si no se puede determinar."
        ),
    )
    chol: Optional[float] = None
    fbs: Optional[float] = Field(
        default=None,
        description="Glucosa en ayunas >120 mg/dl. 1=True/Sí, 0=False/No. None si no aparece.",
    )
    restecg: Optional[float] = Field(
        default=None,
        description=(
            "Resultado electrocardiográfico en reposo. Valores permitidos: "
            "0=normal, 1=ST-T abnormality, 2=left ventricular hypertrophy. "
            "Usar None si no aparece."
        ),
    )
    thalach: Optional[float] = None
    exang: Optional[float] = None
    oldpeak: Optional[float] = None
    slope: Optional[float] = Field(
        default=None,
        description=(
            "Pendiente del segmento ST en ejercicio. Valores permitidos: 0, 1, 2. "
            "Usar None si no aparece."
        ),
    )
    ca: Optional[float] = Field(
        default=None,
        description=(
            "Número de vasos principales coloreados por fluoroscopía. "
            "Valores permitidos: 0, 1, 2, 3. Usar None si no aparece."
        ),
    )
    thal: Optional[float] = Field(
        default=None,
        description=(
            "Variable thal codificada para el modelo cardiovascular. "
            "Valores permitidos: 0=desconocido, 1=normal, 2=fixed defect, 3=reversible defect. "
            "Usar None si no aparece."
        ),
    )

    # Campos clínicos generales
    comorbilidades: List[str] = Field(default_factory=list)
    medicamentos: List[str] = Field(default_factory=list)
    examenes: List[str] = Field(default_factory=list)
    datos_faltantes: List[str] = Field(default_factory=list)