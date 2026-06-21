from pathlib import Path

from langgraph.graph import END, START, StateGraph

from utils.nodes import (
    nodo_analizar,
    nodo_decidir_herramienta,
    nodo_juez,
    nodo_preprocesar_caso,
    nodo_respuesta_final,
    nodo_tool_calcular_map,
    nodo_tool_search_rag,
    nodo_tool_clasificar_paciente_
)
from utils.routers import router_herramientas, router_juez
from utils.state import EstadoClinico





# =========================================================
# 1. CONSTRUIR GRAFO
# =========================================================

def construir_grafo():
    graph = StateGraph(EstadoClinico)

    # Nodos
    graph.add_node("preprocesar", nodo_preprocesar_caso)
    graph.add_node("decidir_herramienta", nodo_decidir_herramienta)
    graph.add_node("calcular_map", nodo_tool_calcular_map)
    graph.add_node("search_rag", nodo_tool_search_rag)
    graph.add_node("clasificar_paciente_cardiovascular", nodo_tool_clasificar_paciente_,)
    graph.add_node("analizar", nodo_analizar)
    graph.add_node("juez", nodo_juez)
    graph.add_node("respuesta_final", nodo_respuesta_final)

    # Flujo base
    graph.add_edge(START, "preprocesar")
    graph.add_edge("preprocesar", "decidir_herramienta")

    # El investigador decide si usa herramienta o analiza
    graph.add_conditional_edges(
        "decidir_herramienta",
        router_herramientas,
        {
            "calcular_map": "calcular_map",
            "search_rag": "search_rag",
            "clasificar_paciente_cardiovascular": "clasificar_paciente_cardiovascular",
            "analizar": "analizar",
        },
    )

    # Después de cada herramienta vuelve al investigador
    graph.add_edge("calcular_map", "decidir_herramienta")
    graph.add_edge("search_rag", "decidir_herramienta")
    graph.add_edge("clasificar_paciente_cardiovascular", "decidir_herramienta")

    # Análisis y revisión
    graph.add_edge("analizar", "juez")

    # El juez puede aprobar o devolver al investigador
    graph.add_conditional_edges(
        "juez",
        router_juez,
        {
            "volver_investigador": "decidir_herramienta",
            "respuesta_final": "respuesta_final",
        },
    )

    graph.add_edge("respuesta_final", END)

    return graph.compile()


# =========================================================
# 2. EJECUTAR GRAFO
# =========================================================

def estado_inicial_desde_caso(caso_clinico: str) -> EstadoClinico:
    return {
        "caso_clinico": caso_clinico,
        "datos_extraidos": {},
        "map_value": None,
        "interpretacion_map": "",
        "query_rag": "",
        "evidencia_rag": [],
        "herramienta_siguiente": "",
        "justificacion_herramienta": "",
        "herramientas_usadas": [],
        "analisis": "",
        "revision": "",
        "observacion_juez": "",
        "respuesta_final": "",
        "ciclos_revision": 0,
        "max_ciclos_revision": 2,
        "intentos_rag": 0,
        "iteraciones_herramientas": 0,
        "max_iteraciones_herramientas": 4,
        "prediccion_cardiovascular_ml": {},
    }


def guardar_diagrama(app) -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    mermaid_path = output_dir / "diagrama_multiagente.mmd"
    mermaid_path.write_text(app.get_graph().draw_mermaid(), encoding="utf-8")
    print(f"\nDiagrama Mermaid guardado en: {mermaid_path.resolve()}")

    try:
        png_path = output_dir / "diagrama_multiagente.png"
        png_path.write_bytes(app.get_graph().draw_mermaid_png())
        print(f"Diagrama PNG guardado en: {png_path.resolve()}")
    except Exception as exc:
        print("No se pudo generar PNG automáticamente.")
        print(f"Usa el .mmd en Mermaid Live Editor. Detalle: {exc}")


def main():
    print("Construyendo grafo ...") #prints de depuracion
    app = construir_grafo()
    print("Grafo construido correctamente") #prints de depuracion

    caso = "Paciente de 70 años con presión arterial 85/55, mareos y antecedente de hipertensión."
    estado_inicial = estado_inicial_desde_caso(caso)

    print("\nINICIANDO SISTEMA MULTIAGENTE CLÍNICO\n") #prints de depuracion

    ultimo_estado = None
    for step in app.stream(estado_inicial, stream_mode="values", config={"recursion_limit": 12}):
        ultimo_estado = step
        print("\n----- ESTADO ACTUAL -----")
        print(step)

    guardar_diagrama(app)

    print("\n----- RESPUESTA FINAL -----")
    print(ultimo_estado["respuesta_final"])


if __name__ == "__main__":
    main()
