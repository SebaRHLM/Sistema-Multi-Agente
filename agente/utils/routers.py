from typing import Literal
from utils.state import EstadoClinico

UMBRAL_CARACTERES_EVIDENCIA_RAG = 4000
MAX_ITEMS_EVIDENCIA_RAG_SIN_RESUMEN = 2
MAX_RESUMENES_RAG = 2

def router_memoria_rag(
    state: EstadoClinico,
) -> Literal["resumir_evidencia_rag", "gestor_clinico"]:
    """
    Router de memoria para evitar que evidencia_rag crezca demasiado.

    Se ejecuta después de search_rag.
    Si la evidencia recuperada es demasiado grande, deriva al nodo resumidor.
    Si no, vuelve al investigador.
    """

    evidencia = state.get("evidencia_rag", [])
    total_caracteres = sum(len(str(item)) for item in evidencia)

    muchas_evidencias = len(evidencia) > MAX_ITEMS_EVIDENCIA_RAG_SIN_RESUMEN
    evidencia_muy_larga = total_caracteres > UMBRAL_CARACTERES_EVIDENCIA_RAG
    resumenes_disponibles = state.get("conteo_resumenes_rag", 0) < MAX_RESUMENES_RAG #Criterio de parada de acuerdo al tope actual de iteraciones del prototipo sin usar llamadas reales de APIs

    if resumenes_disponibles and (muchas_evidencias or evidencia_muy_larga):
        print("[Router Memoria RAG] Evidencia RAG grande. Se deriva al nodo resumidor.")
        return "resumir_evidencia_rag"

    print("[Router Memoria RAG] Evidencia RAG dentro del límite. Se vuelve al investigador.")
    return "gestor_clinico"

def router_herramientas(
    state: EstadoClinico,
) -> Literal["calcular_map", "search_rag", "clasificar_paciente_cardiovascular", "analizar"]:
    """Router controlado por la decisión escrita por el LLM investigador."""

    decision = state["herramienta_siguiente"]

    if decision == "calcular_map":
        return "calcular_map"

    if decision == "search_rag":
        return "search_rag"
    
    if decision == "clasificar_paciente_cardiovascular":
        return "clasificar_paciente_cardiovascular"
    
    # Condición de parada ante alucionaciones
    if decision not in {
        "calcular_map",
        "search_rag",
        "clasificar_paciente_cardiovascular",
        "analizar",
    }:
        print(f"[Router Herramientas] Decisión inválida: {decision}. Se redirige a analizar.")
        return "analizar"

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
