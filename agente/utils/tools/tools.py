from langchain_core.tools import tool
from .search import search, format_search_results


@tool
def calcular_MAP(systolic: int, diastolic: int) -> dict:
    """
    Calcula la presión arterial media, MAP, a partir de la presión sistólica y diastólica.

    Args:
        systolic: Presión arterial sistólica en mmHg.
        diastolic: Presión arterial diastólica en mmHg.

    Returns:
        Diccionario con presión sistólica, diastólica, valor MAP e interpretación general.
    """
    if systolic <= 0 or diastolic <= 0:
        raise ValueError("Los valores de presión arterial deben ser mayores a 0.")

    if systolic <= diastolic:
        raise ValueError("La presión sistólica debe ser mayor que la presión diastólica.")

    map_value = (2 * diastolic + systolic) / 3

    if map_value < 65:
        interpretation = "MAP bajo; posible compromiso de perfusión si hay síntomas o contexto crítico."
    else:
        interpretation = "MAP dentro de rango usualmente aceptable según umbral general."

    return {
        "sistolica": systolic,
        "diastolica": diastolic,
        "map": round(map_value, 2),
        "interpretacion": interpretation,
    }


@tool
def search_clinical_evidence(query: str) -> str:
    """
    Busca evidencia clínica en el RAG sobre hipertensión, hipotensión,
    trastornos de presión arterial y enfermedades cardiovasculares asociadas.
    """
    results = search(query=query, top_k=5)
    return format_search_results(results)

@tool
def clasificar_paciente_cardiovascular(nada:str) -> str:
    """
    Clasifica si el paciente presenta o no una enfermedad cardivascular.
    La clasificación es realizada por un modelo de ML preentrenado por el 
    dataset: 
    """
