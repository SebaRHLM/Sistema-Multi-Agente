from pathlib import Path

from langgraph.graph import END, START, StateGraph

from utils.nodes import (
    nodo_extractor,
    nodo_analizar,
    nodo_decidir_herramienta,
    nodo_juez,
    nodo_respuesta_final,
    nodo_tool_calcular_map,
    nodo_tool_search_rag,
    nodo_tool_clasificar_paciente_,
    nodo_resumir_evidencia_rag
)
from utils.routers import router_herramientas, router_juez, router_memoria_rag
from utils.state import EstadoClinico
from utils.casos__clinicos import casos_de_prueba
from utils.tools.search import inicializar_search_tool





# =========================================================
# 1. CONSTRUIR GRAFO
# =========================================================

def construir_grafo():
    graph = StateGraph(EstadoClinico)

    # Nodos
    graph.add_node("extractor_LLM", nodo_extractor)
    graph.add_node("gestor_clinico", nodo_decidir_herramienta)
    graph.add_node("calcular_map", nodo_tool_calcular_map)
    graph.add_node("search_rag", nodo_tool_search_rag)
    graph.add_node("resumir_evidencia_rag", nodo_resumir_evidencia_rag)
    graph.add_node("clasificar_paciente_cardiovascular", nodo_tool_clasificar_paciente_,)
    graph.add_node("analizar", nodo_analizar)
    graph.add_node("juez", nodo_juez)
    graph.add_node("respuesta_final", nodo_respuesta_final)

    # Flujo base
    graph.add_edge(START, "extractor_LLM")
    graph.add_edge("extractor_LLM", "gestor_clinico")

    # El investigador decide si usa herramienta o analiza
    graph.add_conditional_edges(
        "gestor_clinico",
        router_herramientas,
        {
            "calcular_map": "calcular_map",
            "search_rag": "search_rag",
            "clasificar_paciente_cardiovascular": "clasificar_paciente_cardiovascular",
            "analizar": "analizar",
        },
    )

    # Después de cada herramienta vuelve al investigador
    graph.add_edge("calcular_map", "gestor_clinico")
    graph.add_edge("clasificar_paciente_cardiovascular", "gestor_clinico")
    
    # Decidir si resumir o volver al investigador
    graph.add_conditional_edges(
        "search_rag",
        router_memoria_rag,
        {
            "resumir_evidencia_rag": "resumir_evidencia_rag",
            "gestor_clinico": "gestor_clinico",
        },
    )
    graph.add_edge("resumir_evidencia_rag", "gestor_clinico")

    # Análisis y revisión
    graph.add_edge("analizar", "juez")

    # El juez puede aprobar o devolver al investigador
    graph.add_conditional_edges(
        "juez",
        router_juez,
        {
            "volver_investigador": "gestor_clinico",
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
        "resumen_evidencia_rag": "",
        "evidencia_rag_resumida": False,
        "caracteres_evidencia_rag": 0,
        "conteo_resumenes_rag": 0,  
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

def seleccionar_caso_clinico() -> str:
    """
    Permite seleccionar manualmente un caso clínico de prueba del 1 al 3.
    Retorna el texto del caso seleccionado.
    """

    print("\nCASOS CLÍNICOS DISPONIBLES")
    print("1. Caso 1: Dolor torácico + ECG compatible con evento coronario")
    print("2. Caso 2: Diálisis + CKD + diabetes + fiebre + disnea")
    print("3. Caso 3: Marcapasos + fibrilación auricular + insuficiencia cardíaca")

    while True:
        opcion = input("\nSelecciona un caso clínico [1-3]: ").strip()

        if opcion in ["1", "2", "3"]:
            indice = int(opcion) - 1
            caso = casos_de_prueba[indice]

            print(f"\nCaso clínico seleccionado: {opcion}")
            return caso

        print("Opción inválida. Debes ingresar 1, 2 o 3.")

def precargar_rag():
    """
    Carga el retriever RAG antes de iniciar el flujo del grafo.
    Esto evita que la primera llamada a search_rag bloquee el nodo durante la demo.
    """
    print("[Startup] Inicializando RAG: ChromaDB + SentenceTransformer ...")

    try:
        inicializar_search_tool()
        print("[Startup] RAG inicializado correctamente.")
    except Exception as exc:
        print(f"[Startup] No se pudo inicializar el RAG: {exc}")
        print("[Startup] El sistema continuará, pero search_rag podría fallar o usar fallback.")


def main():
    print("Construyendo grafo ...") #prints de depuracion
    app = construir_grafo()
    print("Grafo construido correctamente") #prints de depuracion

    print("Cargando base vectorial y embedder para tool rag")
    precargar_rag()
    print("base vectorial y embedder cargados correctamente")

    caso = seleccionar_caso_clinico()
    estado_inicial = estado_inicial_desde_caso(caso)

    print("\nINICIANDO SISTEMA MULTIAGENTE CLÍNICO\n") #prints de depuracion

    ultimo_estado = None
    for step in app.stream(estado_inicial, stream_mode="values", config={"recursion_limit": 20}):
        ultimo_estado = step
        print("\n----- ESTADO ACTUAL -----")
        print(step)

    guardar_diagrama(app)

    print("\n----- RESPUESTA FINAL -----")
    print(ultimo_estado["respuesta_final"])


if __name__ == "__main__":
    main()
