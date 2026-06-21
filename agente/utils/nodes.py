import re
from typing import Optional
import json

from langchain_core.prompts import ChatPromptTemplate

from utils.llms import USE_REAL_LLM, llm_investigador, llm_juez
from utils.schemas import DecisionHerramienta, RevisionJuez
from utils.state import EstadoClinico
from utils.tools.tools import calcular_MAP, search_clinical_evidence, clasificar_paciente_cardiovascular


# =========================================================
# Helpers internos
# =========================================================


def _extraer_presion(texto: str) -> tuple[Optional[int], Optional[int]]:
    """Extrae una presiÃ³n tipo 85/55 desde texto natural."""
    match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", texto)
    if not match:
        return None, None

    return int(match.group(1)), int(match.group(2))


def _extraer_sintomas_basicos(texto: str) -> list[str]:
    sintomas_posibles = [
        "mareo",
        "debilidad",
        "dolor torÃ¡cico",
        "disnea",
        "sincope",
        "cefalea",
        "vision borrosa",
        "palpitaciones",
        "sudoracion",
    ]
    texto_lower = texto.lower()
    return [s for s in sintomas_posibles if s in texto_lower]


def _extraer_antecedentes_basicos(texto: str) -> list[str]:
    antecedentes_posibles = [
        "hipertension",
        "diabetes",
        "enfermedad renal cronica",
        "insuficiencia cardiaca",
        "coronaria",
        "tabaquismo",
    ]
    texto_lower = texto.lower()
    return [a for a in antecedentes_posibles if a in texto_lower]

def _extraer_edad(texto: str) -> Optional[int]:
    match = re.search(r"(\d{1,3})\s*(años|year|years)", texto.lower())
    if match:
        return int(match.group(1))
    return None


def _extraer_sexo(texto: str) -> Optional[int]:
    texto = texto.lower()
    if any(x in texto for x in ["masculino", "hombre", "male"]):
        return 1
    if any(x in texto for x in ["femenino", "mujer", "female"]):
        return 0
    return None


# =========================================================
# Nodo 1: Preprocesamiento
# =========================================================


def nodo_preprocesar_caso(state: EstadoClinico):
    print("[Nodo] Preprocesar caso clinico")

    caso = state["caso_clinico"]
    sistolica, diastolica = _extraer_presion(caso)

    datos = {
        "sistolica": sistolica,
        "diastolica": diastolica,
        "trestbps": sistolica,
        "age": _extraer_edad(caso),
        "sex": _extraer_sexo(caso),
        "sintomas": _extraer_sintomas_basicos(caso),
        "antecedentes": _extraer_antecedentes_basicos(caso),
    }

    return {"datos_extraidos": datos}


# =========================================================
# Nodo 2: LLM Investigador decide herramienta
# =========================================================


def _decision_mock_investigador(state: EstadoClinico):
    herramientas_usadas = state["herramientas_usadas"]
    datos = state["datos_extraidos"]

    # Corte defensivo: si el juez ya rechazÃ³ varias veces, no sigas analizando.
    if state["ciclos_revision"] >= state["max_ciclos_revision"]:
        return DecisionHerramienta(
            herramienta_siguiente="analizar",
            justificacion_herramienta="Se alcanzÃ³ el mÃ¡ximo de ciclos de revisiÃ³n; se generarÃ¡ una salida con limitaciones."
        )

    # Si falta MAP y hay datos de presiÃ³n, calcular MAP.
    if (
        datos.get("sistolica") is not None
        and datos.get("diastolica") is not None
        and "calcular_map" not in herramientas_usadas
    ):
        return DecisionHerramienta(
            herramienta_siguiente="calcular_map",
            justificacion_herramienta="El caso contiene presiÃ³n sistÃ³lica y diastÃ³lica; se requiere calcular MAP."
        )

    # Si no hay evidencia real, buscar RAG.
    if (
        len(state["evidencia_rag"]) == 0
        and state["intentos_rag"] < 2
        and "search_rag" not in herramientas_usadas
    ):
        return DecisionHerramienta(
            herramienta_siguiente="search_rag",
            justificacion_herramienta="Se requiere evidencia clÃ­nica para fundamentar el anÃ¡lisis."
        )

    # Si el juez detecta falta de evidencia y aun queda intento RAG.
    if (
        state["revision"] == "falta_evidencia"
        and state["intentos_rag"] < 2
    ):
        return DecisionHerramienta(
            herramienta_siguiente="search_rag",
            justificacion_herramienta="El juez indica falta de evidencia; se debe realizar una nueva busqueda RAG."
        )
    
    if state["revision"] == "falta_datos":
        return DecisionHerramienta(
            herramienta_siguiente="analizar",
            justificacion_herramienta=(
                "El juez detectó falta de datos. No existe una herramienta capaz de inventarlos; "
                "se generará salida con limitaciones y datos faltantes explícitos."
            ),
    )

    # En cualquier otro caso, analizar.
    return DecisionHerramienta(
        herramienta_siguiente="analizar",
        justificacion_herramienta="El estado contiene datos suficientes para generar un analisis preliminar o una salida con limitaciones."
    )


def nodo_decidir_herramienta(state: EstadoClinico):
    print("[LLM Investigador] Decidiendo si usar herramienta")

    if not USE_REAL_LLM:
        decision = _decision_mock_investigador(state)
    else:
        llm_estructurado = llm_investigador.with_structured_output(DecisionHerramienta)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Eres un agente investigador clinico dentro de un sistema de apoyo mÃ©dico.
                    Tu tarea es decidir cuÃ¡l es el siguiente recurso que debe usar el sistema.

                    Opciones disponibles:
                    - calcular_map: usar si el caso contiene presiÃ³n sistÃ³lica y diastÃ³lica y todavia no se calculÃ³ MAP.
                    - search_rag: usar si falta evidencia clinica para fundamentar el anÃ¡lisis.
                    - analizar: usar si ya hay datos y evidencia suficientes para elaborar un anÃ¡lisis preliminar.

                    Restricciones:
                    - Decide solo una acciÃ³n.
                    - No entregues diagnÃ³stico definitivo.
                    - No recomiendes prescripcion farmacolÃ³gica automÃ¡tica.
                    """,
                ),
                (
                    "user",
                    """
                    Caso clinico: {caso_clinico}
                    Datos extraidos: {datos_extraidos}
                    MAP actual: {map_value}
                    Evidencia RAG: {evidencia_rag}
                    Herramientas usadas: {herramientas_usadas}
                    Intentos RAG: {intentos_rag}
                    """,
                ),
            ]
        )

        chain = prompt | llm_estructurado
        decision = chain.invoke(
            {
                "caso_clinico": state["caso_clinico"],
                "datos_extraidos": state["datos_extraidos"],
                "map_value": state["map_value"],
                "evidencia_rag": state["evidencia_rag"],
                "herramientas_usadas": state["herramientas_usadas"],
                "intentos_rag": state["intentos_rag"],
            }
        )

    print(f"[LLM Investigador] DecisiÃ³n: {decision.herramienta_siguiente}")
    print(f"[LLM Investigador] JustificaciÃ³n: {decision.justificacion_herramienta}")

    return {
        "herramienta_siguiente": decision.herramienta_siguiente,
        "justificacion_herramienta": decision.justificacion_herramienta,
    }


# =========================================================
# Nodo herramienta 1: calcular MAP
# =========================================================


def nodo_tool_calcular_map(state: EstadoClinico):
    print("[Tool] calcular_MAP")

    datos = state["datos_extraidos"]
    sistolica = datos.get("sistolica")
    diastolica = datos.get("diastolica")

    if sistolica is None or diastolica is None:
        return {
            "map_value": None,
            "interpretacion_map": "No se pudo calcular MAP porque faltan presiÃ³n sistÃ³lica o diastÃ³lica.",
            "herramientas_usadas": ["calcular_map"],
            "iteraciones_herramientas": state["iteraciones_herramientas"] + 1,
        }

    resultado = calcular_MAP.invoke(
        {
            "systolic": sistolica,
            "diastolic": diastolica,
        }
    )

    return {
        "map_value": resultado["map"],
        "interpretacion_map": resultado["interpretacion"],
        "herramientas_usadas": ["calcular_map"],
        "iteraciones_herramientas": state["iteraciones_herramientas"] + 1,
    }


# =========================================================
# Nodo herramienta 2: RAG
# =========================================================


def nodo_tool_search_rag(state: EstadoClinico):
    print("[Tool] search_clinical_evidence")

    datos = state["datos_extraidos"]
    query = f"""
    Trastornos de presion arterial y enfermedades cardiovasculares asociadas.
    Presion arterial: {datos.get('sistolica')}/{datos.get('diastolica')}.
    Sintomas: {datos.get('sintomas')}.
    Antecedentes: {datos.get('antecedentes')}.
    Buscar evidencia sobre hipotension, hipertension, MAP,
    evaluacion clinica y diagnosticos diferenciales.
    """.strip()

    try:
        evidencia = search_clinical_evidence.invoke({"query": query})
    except Exception as exc:
        evidencia = (
            "Evidencia RAG simulada por error o base no inicializada: "
            "se deben evaluar signos, sintomas, comorbilidades y causas cardiovasculares. "
            f"Detalle tecnico: {exc}"
        )

    return {
        "query_rag": query,
        "evidencia_rag": [evidencia],
        "herramientas_usadas": ["search_rag"],
        "intentos_rag": state["intentos_rag"] + 1,
        "iteraciones_herramientas": state["iteraciones_herramientas"] + 1,
    }

# =========================================================
# Nodo herramienta 3: clasificar paciente
# =========================================================

def nodo_tool_clasificar_paciente_ (state: EstadoClinico):
    print("[Tool] clasificar_paciente_cardiovascular")

    datos = state["datos_extraidos"]

    resultado_str = clasificar_paciente_cardiovascular.invoke({
        "age": datos.get("age"),
        "sex": datos.get("sex"),
        "cp": datos.get("cp"),
        "trestbps": datos.get("trestbps"),
        "chol": datos.get("chol"),
        "fbs": datos.get("fbs"),
        "restecg": datos.get("restecg"),
        "thalach": datos.get("thalach"),
        "exang": datos.get("exang"),
        "oldpeak": datos.get("oldpeak"),
        "slope": datos.get("slope"),
        "ca": datos.get("ca"),
        "thal": datos.get("thal"),
    })

    try:
        resultado = json.loads(resultado_str)
    except Exception:
        resultado = {
            "estado": "error_parseo",
            "respuesta_cruda": resultado_str,
        }

    return {
        "prediccion_cardiovascular_ml": resultado,
        "herramientas_usadas": ["clasificar_paciente_cardiovascular"],
        "iteraciones_herramientas": state["iteraciones_herramientas"] + 1,
    }

# =========================================================
# Nodo LLM Investigador: analisis preliminar
# =========================================================


def nodo_analizar(state: EstadoClinico):
    print("[LLM Analizador] Generando análisis preliminar")

    if not USE_REAL_LLM:
        analisis = f"""
        El caso describe un paciente con datos compatibles con trastorno de presiÃ³n arterial.
        Datos extraidos: {state['datos_extraidos']}.
        MAP calculado: {state['map_value']}.
        Interpretacion MAP: {state['interpretacion_map']}.
        Evidencia recuperada: {state['evidencia_rag']}.

        Analisis preliminar:
        - Considerar hipotension sintomatica si la presion baja se asocia a mareos, sincope, debilidad o signos de hipoperfusion.
        - Evaluar contexto cardiovascular, comorbilidades y necesidad de examenes complementarios.
        - La respuesta debe ser interpretada por un profesional medico y no corresponde a diagnastico definitivo.
        """.strip()
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Eres un agente investigador clinico. Genera un anÃ¡lisis preliminar util para un medico.
                    Usa solo los datos del caso y la evidencia entregada.
                    No entregues diagnostico definitivo ni prescripcion automatica.
                    """,
                ),
                (
                    "user",
                    """
                    Caso clinico: {caso_clinico}
                    Datos extraidos: {datos_extraidos}
                    MAP: {map_value}
                    Interpretacion MAP: {interpretacion_map}
                    Predicción cardiovascular ML: {prediccion_cardiovascular_ml}
                    Evidencia RAG: {evidencia_rag}
                    """,
                ),
            ]
        )
        chain = prompt | llm_investigador
        respuesta = chain.invoke(
            {
                "caso_clinico": state["caso_clinico"],
                "datos_extraidos": state["datos_extraidos"],
                "map_value": state["map_value"],
                "interpretacion_map": state["interpretacion_map"],
                "evidencia_rag": state["evidencia_rag"],
                "prediccion_cardiovascular_ml": state["prediccion_cardiovascular_ml"],
            }
        )
        analisis = respuesta.content

    return {"analisis": analisis}


# =========================================================
# Nodo LLM Juez
# =========================================================


def _revision_mock_juez(state: EstadoClinico) -> RevisionJuez:
    if len(state["evidencia_rag"]) == 0:
        estado_revision = "falta_evidencia"
        observacion = "No existe evidencia clinica recuperada desde el RAG."
    elif "Evidencia RAG simulada por error" in str(state["evidencia_rag"]):
        estado_revision = "falta_evidencia"
        observacion = "La evidencia proviene de un fallback simulado por error del RAG."
    elif "diagnostico definitivo" in state["analisis"].lower() or "diagnÃ³stico definitivo" in state["analisis"].lower():
        estado_revision = "riesgo_alucinacion"
        observacion = "El analisis contiene lenguaje que podria interpretarse como diagnostico definitivo."
    else:
        estado_revision = "aprobado"
        observacion = "La respuesta esta suficientemente fundamentada para apoyo clinico."

    print(f"[LLM Juez] Revision: {estado_revision}")
    print(f"[LLM Juez] Observacion: {observacion}")

    return RevisionJuez(estado_revision=estado_revision, observacion=observacion)

def nodo_juez(state: EstadoClinico):
    print("[LLM Juez] Evaluando suficiencia de la informaciÃ³n")

    if not USE_REAL_LLM:
        revision = _revision_mock_juez(state)
    else:
        llm_estructurado = llm_juez.with_structured_output(RevisionJuez)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Eres un juez clinico de un sistema Multiagente.
                    Evalua si el analisis preliminar es suficiente, esta fundamentado y respeta el alcance.

                    Debes revisar:
                    - datos clinicos minimos;
                    - evidencia RAG suficiente;
                    - coherencia entre evidencia y analisis;
                    - ausencia de diagnostico definitivo o prescripcion automatica.
                    """,
                ),
                (
                    "user",
                    """
                    Caso clinico: {caso_clinico}
                    Datos extraidos: {datos_extraidos}
                    MAP: {map_value}
                    Evidencia RAG: {evidencia_rag}
                    Analisis preliminar: {analisis}
                    """,
                ),
            ]
        )

        chain = prompt | llm_estructurado
        revision = chain.invoke(
            {
                "caso_clinico": state["caso_clinico"],
                "datos_extraidos": state["datos_extraidos"],
                "map_value": state["map_value"],
                "evidencia_rag": state["evidencia_rag"],
                "analisis": state["analisis"],
            }
        )

    print(f"[LLM Juez] RevisiÃ³n: {revision.estado_revision}")
    print(f"[LLM Juez] ObservaciÃ³n: {revision.observacion}")

    return {
        "revision": revision.estado_revision,
        "observacion_juez": revision.observacion,
        "ciclos_revision": state["ciclos_revision"] + 1,
    }


# =========================================================
# Nodo final
# =========================================================


def nodo_respuesta_final(state: EstadoClinico):
    print("[Nodo-Respuesta Final] Redactando respuesta final") #prints de depuracion

    if state["revision"] != "aprobado":
        respuesta = f"""
        REPORTE DE APOYO CLÃNICO CON LIMITACIONES

        Caso ingresado:
        {state["caso_clinico"]}

        Datos extraÃ­dos:
        {state["datos_extraidos"]}

        MAP:
        {state["map_value"]}

        InterpretaciÃ³n MAP:
        {state["interpretacion_map"]}

        AnÃ¡lisis preliminar:
        {state["analisis"]}

        Estado de revisiÃ³n:
        {state["revision"]}

        ObservaciÃ³n del juez:
        {state["observacion_juez"]}

        LimitaciÃ³n:
        El sistema no logrÃ³ una aprobaciÃ³n completa del juez dentro del mÃ¡ximo de ciclos permitido.
        Esta salida debe interpretarse solo como apoyo preliminar y requiere revisiÃ³n del profesional mÃ©dico.

        Nota:
        El sistema no entrega diagnÃ³stico definitivo ni prescribe tratamiento.
        """
    else:
        respuesta = f"""
        REPORTE DE APOYO CLÃNICO

        Caso ingresado:
        {state["caso_clinico"]}

        Datos extraÃ­dos:
        {state["datos_extraidos"]}

        MAP:
        {state["map_value"]}

        InterpretaciÃ³n MAP:
        {state["interpretacion_map"]}

        Evidencia usada:
        {state["evidencia_rag"]}

        AnÃ¡lisis:
        {state["analisis"]}

        Nota:
        El sistema entrega apoyo al razonamiento clÃ­nico. No reemplaza el criterio profesional mÃ©dico.
        """

    return {"respuesta_final": respuesta.strip()}
