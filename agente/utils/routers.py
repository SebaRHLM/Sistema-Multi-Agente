from typing import Literal
from utils.state import EstadoClinico


def router_herramientas(
    state: EstadoClinico,
) -> Literal["calcular_map", "search_rag", "analizar"]:
    """Router controlado por la decisión escrita por el LLM investigador."""

    decision = state["herramienta_siguiente"]

    if decision == "calcular_map":
        return "calcular_map"

    if decision == "search_rag":
        return "search_rag"

    return "analizar"


def router_juez(
    state: EstadoClinico,
) -> Literal["volver_investigador", "respuesta_final"]:

    revision = state["revision"]

    ciclos_revision = state["ciclos_revision"]
    max_ciclos_revision = state["max_ciclos_revision"]

    iteraciones_herramientas = state["iteraciones_herramientas"]
    max_iteraciones_herramientas = state["max_iteraciones_herramientas"]

    # Si el juez aprueba, termina.
    if revision == "aprobado":
        return "respuesta_final"

    # Corte duro por ciclos de revisión.
    if ciclos_revision >= max_ciclos_revision:
        print("[Router Juez] Máximo de ciclos de revisión alcanzado. Se finaliza con limitaciones.")
        return "respuesta_final"

    # Corte duro por herramientas.
    if iteraciones_herramientas >= max_iteraciones_herramientas:
        print("[Router Juez] Máximo de herramientas alcanzado. Se finaliza con limitaciones.")
        return "respuesta_final"

    # Si todavía puede corregir, vuelve al investigador.
    if revision in ["falta_evidencia", "falta_datos", "riesgo_alucinacion"]:
        return "volver_investigador"

    return "respuesta_final"
