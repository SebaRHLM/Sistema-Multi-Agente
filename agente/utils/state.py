from typing import Annotated, Dict, List, Optional, TypedDict
import operator


class EstadoClinico(TypedDict):
    # Entrada original
    caso_clinico: str

    # Preprocesamiento clínico
    datos_extraidos: Dict
    map_value: Optional[float]
    interpretacion_map: str

    # RAG
    query_rag: str
    evidencia_rag: Annotated[List[str], operator.add]

    # Decisión del investigador
    herramienta_siguiente: str
    justificacion_herramienta: str
    herramientas_usadas: Annotated[List[str], operator.add]

    # Análisis y revisión
    analisis: str
    revision: str
    observacion_juez: str
    respuesta_final: str

    # Control de ciclos
    ciclos_revision: int
    max_ciclos_revision: int
    intentos_rag: int
    iteraciones_herramientas: int
    max_iteraciones_herramientas: int
