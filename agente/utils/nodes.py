import re
from typing import Optional
import json

from langchain_core.prompts import ChatPromptTemplate

from utils.llms import USE_REAL_LLM, llm_openai
from utils.schemas import DecisionHerramienta, RevisionJuez, DatosExtraidos_LLM_Extractor
from utils.state import EstadoClinico
from utils.tools.tools import calcular_MAP, search_clinical_evidence, clasificar_paciente_cardiovascular


# =========================================================
# Helpers internos
# =========================================================


# Deprecated
def _extraer_presion(texto: str) -> tuple[Optional[int], Optional[int]]:
    """Extrae una presiÃ³n tipo 85/55 desde texto natural."""
    match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", texto)
    if not match:
        return None, None

    return int(match.group(1)), int(match.group(2))


# Deprecated
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


# Deprecated
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

# Deprecated
def _extraer_edad(texto: str) -> Optional[int]:
    match = re.search(r"(\d{1,3})\s*(años|year|years)", texto.lower())
    if match:
        return int(match.group(1))
    return None

# Deprecated
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

# Deprecated
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
# Nodo 1: Extractor basado en LLM
# =========================================================

# mock generado por IA para depuracion del grafo.
def _extraer_presion_arterial_mock(texto: str):
    """
    Extrae presión arterial en formato 85/55, 120/80 mmHg, etc.
    """
    match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", texto)

    if not match:
        return None, None

    sistolica = int(match.group(1))
    diastolica = int(match.group(2))

    return sistolica, diastolica


def _extraer_edad_mock(texto: str):
    """
    Extrae edad desde expresiones como:
    - Paciente de 70 años
    - A 49-year-old patient
    - 57-year-old male patient
    """
    patrones = [
        r"(\d{1,3})\s*años",
        r"(\d{1,3})\s*year-old",
        r"(\d{1,3})\s*years old",
    ]

    texto_lower = texto.lower()

    for patron in patrones:
        match = re.search(patron, texto_lower)
        if match:
            return int(match.group(1))

    return None


def _extraer_sexo_mock(texto: str):
    """
    Codificación compatible con el modelo cardiovascular:
    1 = masculino
    0 = femenino
    None = no informado
    """
    texto_lower = texto.lower()

    if any(x in texto_lower for x in ["masculino", "hombre", "male"]):
        return 1

    if any(x in texto_lower for x in ["femenino", "mujer", "female"]):
        return 0

    return None


def _extraer_sintomas_mock(texto: str):
    texto_lower = texto.lower()

    patrones_sintomas = {
        "mareo": ["mareo", "mareos", "dizziness", "lightheadedness"],
        "síncope": ["síncope", "sincope", "syncope"],
        "dolor torácico": ["dolor torácico", "dolor toracico", "chest pain"],
        "disnea": ["disnea", "shortness of breath", "dyspnea"],
        "debilidad": ["debilidad", "weakness"],
        "sudoración": ["sudoración", "sudoracion", "sweating"],
        "ortopnea": ["orthopnea", "ortopnea"],
        "edema periférico": ["peripheral edema", "edema periférico", "edema periferico"],
    }

    sintomas = []

    for sintoma, claves in patrones_sintomas.items():
        if any(clave in texto_lower for clave in claves):
            sintomas.append(sintoma)

    return sintomas


def _extraer_antecedentes_mock(texto: str):
    texto_lower = texto.lower()

    patrones_antecedentes = {
        "hipertensión": ["hipertensión", "hipertension", "hypertension", "hta"],
        "diabetes mellitus": ["diabetes", "diabetes mellitus", "dm", "type 2 diabetes"],
        "enfermedad renal crónica": [
            "enfermedad renal crónica",
            "enfermedad renal cronica",
            "chronic kidney disease",
            "ckd",
            "chronic kidney failure",
        ],
        "tabaquismo": ["tabaquismo", "fumador", "smoking", "smoker"],
        "fibrilación auricular": [
            "fibrilación auricular",
            "fibrilacion auricular",
            "atrial fibrillation",
            "af",
        ],
        "insuficiencia cardíaca": [
            "insuficiencia cardíaca",
            "insuficiencia cardiaca",
            "heart failure",
        ],
    }

    antecedentes = []

    for antecedente, claves in patrones_antecedentes.items():
        if any(clave in texto_lower for clave in claves):
            antecedentes.append(antecedente)

    return antecedentes


def _extraer_medicamentos_mock(texto: str):
    texto_lower = texto.lower()

    patrones_medicamentos = {
        "betabloqueador": ["beta blocker", "beta blockers", "betabloqueador", "betabloqueadores"],
        "anticoagulante": ["anticoagulant", "anticoagulante"],
        "amlodipino": ["amlodipine", "amlodipino"],
        "furosemida": ["furosemide", "furosemida"],
        "insulina": ["insulin", "insulina"],
    }

    medicamentos = []

    for medicamento, claves in patrones_medicamentos.items():
        if any(clave in texto_lower for clave in claves):
            medicamentos.append(medicamento)

    return medicamentos


def _extraer_examenes_mock(texto: str):
    texto_lower = texto.lower()
    examenes = []

    if "ecg" in texto_lower or "electrocardiogram" in texto_lower:
        examenes.append("ECG mencionado")

    if "st depression" in texto_lower or "depresión st" in texto_lower or "depresion st" in texto_lower:
        examenes.append("ECG con depresión del ST")

    if "st elevation" in texto_lower or "elevación st" in texto_lower or "elevacion st" in texto_lower:
        examenes.append("ECG con elevación del ST")

    if "echocardiography" in texto_lower or "ecocardiograma" in texto_lower:
        examenes.append("ecocardiograma mencionado")

    if "holter" in texto_lower:
        examenes.append("Holter ECG mencionado")

    return examenes


def _mock_nodo_extractor(state):
    """
    Mock provisional del nodo extractor.

    Devuelve la misma estructura de datos_extraidos que esperan las herramientas actuales.
    No usa LLM real.
    """
    caso = state["caso_clinico"]

    sistolica, diastolica = _extraer_presion_arterial_mock(caso)
    edad = _extraer_edad_mock(caso)
    sexo = _extraer_sexo_mock(caso)

    sintomas = _extraer_sintomas_mock(caso)
    antecedentes = _extraer_antecedentes_mock(caso)
    medicamentos = _extraer_medicamentos_mock(caso)
    examenes = _extraer_examenes_mock(caso)

    datos_faltantes = []

    if sistolica is None or diastolica is None:
        datos_faltantes.append("presión arterial")

    if edad is None:
        datos_faltantes.append("edad")

    if sexo is None:
        datos_faltantes.append("sexo")

    if "frecuencia cardíaca" not in caso.lower() and "heart rate" not in caso.lower():
        datos_faltantes.append("frecuencia cardíaca")

    if not medicamentos:
        datos_faltantes.append("medicamentos actuales")

    datos_extraidos = {
        "sistolica": sistolica,
        "diastolica": diastolica,
        "trestbps": sistolica,

        "age": edad,
        "sex": sexo,

        "sintomas": sintomas,
        "antecedentes": antecedentes,
        "comorbilidades": antecedentes,
        "medicamentos": medicamentos,
        "examenes": examenes,
        "datos_faltantes": datos_faltantes,

        # Variables opcionales del modelo cardiovascular.
        # El mock no las inventa.
        "cp": None,
        "chol": None,
        "fbs": 1 if "diabetes" in caso.lower() else None,
        "restecg": None,
        "thalach": None,
        "exang": None,
        "oldpeak": None,
        "slope": None,
        "ca": None,
        "thal": None,
    }

    print("[Mock Extractor] Datos extraídos:", datos_extraidos)

    return {
        "datos_extraidos": datos_extraidos
    }

def nodo_extractor(state: EstadoClinico):
    """
    Nodo extractor basado en LLM.

    Extrae desde el caso clinico los mismos valores que antes se guardaban
    en datos_extraidos, pero con mejor capacidad para reconocer antecedentes,
    síntomas, signos vitales, medicamentos y variables del modelo ML.
    """

    print("[LLM Extractor] Extrayendo datos clínicos del caso")

    if not USE_REAL_LLM:
        return _mock_nodo_extractor(state)

    if llm_openai is None:
        raise RuntimeError(
            "USE_REAL_LLM=True, pero llm_openai es None. "
            "Revisa OPENAI_API_KEY, OPENAI_MODEL y la configuración de llms.py."
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    Eres un extractor de datos clínicos para un sistema multiagente de apoyo clínico.

                    Tu única tarea es leer un caso clínico y extraer datos estructurados para la variable datos_extraidos.

                    No debes diagnosticar.
                    No debes analizar el caso.
                    No debes proponer tratamiento.
                    No debes inventar datos.

                    Reglas:
                    - Si un dato no aparece explícitamente, usa null.
                    - Si una lista no tiene elementos, usa lista vacía.
                    - Extrae presión arterial si aparece en formato como 85/55 mmHg.
                    - sistolica corresponde al primer valor de presión arterial.
                    - diastolica corresponde al segundo valor de presión arterial.
                    - trestbps debe ser igual a sistolica si se reporta presión arterial.
                    - age corresponde a la edad del paciente.
                    - sex debe ser 1 si el paciente es masculino/hombre/male.
                    - sex debe ser 0 si el paciente es femenino/mujer/female.
                    - Si el sexo no aparece, usa null.
                    - sintomas debe contener síntomas explícitos del caso.
                    - antecedentes debe contener antecedentes médicos explícitos.
                    - comorbilidades puede repetir condiciones clínicas relevantes como diabetes, CKD, insuficiencia cardíaca, fibrilación auricular, hipertensión, etc.
                    - medicamentos debe contener fármacos mencionados.
                    - examenes debe contener hallazgos de ECG, ecocardiograma, laboratorio, imágenes o examen físico relevante.
                    - datos_faltantes debe listar datos clínicos importantes que no aparecen y que serían útiles para evaluar el caso.

                    Variables del modelo cardiovascular:
                    - age: edad del paciente en años.
                    - sex: 1=male/masculino, 0=female/femenino.
                    - cp: tipo de dolor torácico:
                    0=typical angina,
                    1=atypical angina,
                    2=non-anginal,
                    3=asymptomatic.

                    - trestbps: presión arterial sistólica en reposo o al ingreso hospitalario.
                    - chol: colesterol sérico en mg/dl.
                    - fbs: 1 si fasting blood sugar >120 mg/dl, 0 si no.
                    - restecg:
                    0=normal,
                    1=ST-T abnormality,
                    2=left ventricular hypertrophy.

                    - thalach: frecuencia cardíaca máxima alcanzada.
                    - exang: angina inducida por ejercicio, 1=True, 0=False.
                    - oldpeak: depresión ST inducida por ejercicio relativa al reposo.
                    - slope: pendiente del segmento ST en ejercicio, valores permitidos 0, 1, 2.
                    - ca: número de vasos principales coloreados por fluoroscopía, valores permitidos 0, 1, 2, 3.
                    - thal:
                    1=normal,
                    2=fixed defect,
                    3=reversible defect.

                    Reglas:
                    - No inventes variables del modelo.
                    - Si una variable no aparece explícitamente o no puede inferirse de forma segura, usa null.
                """,
            ),
            (
                "user",
                """
                    Caso clínico:{caso_clinico}
                    Extrae los datos clínicos estructurados.
                    """,
                ),
            ]
        )

    chain = prompt | llm_openai.with_structured_output(DatosExtraidos_LLM_Extractor)

    datos = chain.invoke(
        {
            "caso_clinico": state["caso_clinico"],
        }
    )

    datos_dict = datos.model_dump()

    print("[LLM Extractor] Datos extraídos:", datos_dict)

    return {
        "datos_extraidos": datos_dict
    }

# =========================================================
# Nodo LLM 1: Investigador decide herramienta
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
        llm_estructurado = llm_openai.with_structured_output(DecisionHerramienta)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Eres el Gestor Clínico de un sistema multiagente de apoyo clínico.

                    Tu única tarea es decidir cuál es el siguiente nodo o herramienta que debe ejecutar el sistema.
                    No debes generar análisis clínico completo.
                    No debes entregar diagnóstico definitivo.
                    No debes prescribir tratamiento.

                    Opciones disponibles:
                    - calcular_map: usar si existen presión sistólica y diastólica, y todavía no se calculó MAP.
                    - clasificar_paciente_cardiovascular: usar si existen al menos 5 variables compatibles con el modelo cardiovascular:
                    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.
                    - search_rag: usar si falta evidencia clínica para fundamentar el análisis o si el juez indicó falta_evidencia.
                    - analizar: usar si ya hay datos y evidencia suficientes, o si faltan datos que ninguna herramienta puede inferir.

                    Restricciones:
                    - Decide solo una acción.
                    - No repitas herramientas si no aportarán información nueva.
                    - Si ya se alcanzó max_iteraciones_herramientas, selecciona analizar.
                    - No inventes datos faltantes.
                    - No entregues diagnóstico definitivo.
                    - No recomiendes prescripción farmacológica automática.
                    """,
                ),
                (
                    "user",
                    """
                    Caso clinico: {caso_clinico}
                    Datos extraidos: {datos_extraidos}
                    MAP actual: {map_value}
                    Resumen/evidencia RAG disponible: {hay_evidencia_rag}
                    Cantidad de elementos en evidencia_rag: {cantidad_evidencia_rag}
                    Caracteres aproximados de evidencia_rag: {caracteres_evidencia_rag}
                    Evidencia RAG resumida: {evidencia_rag_resumida}
                    Intentos RAG: {intentos_rag}
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
                "hay_evidencia_rag": len(state["evidencia_rag"]) > 0,
                "cantidad_evidencia_rag": len(state["evidencia_rag"]),
                "caracteres_evidencia_rag": state.get("caracteres_evidencia_rag", 0),
                "evidencia_rag_resumida": state.get("evidencia_rag_resumida", False),
                "herramientas_usadas": state["herramientas_usadas"],
                "intentos_rag": state["intentos_rag"],
            }
        )

    print(f"[LLM Investigador] Decision: {decision.herramienta_siguiente}")
    print(f"[LLM Investigador] Justificacion: {decision.justificacion_herramienta}")

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

    evidencia_anterior = state.get("evidencia_rag", [])

    # Acumulación manual controlada sin depender de operator.add.
    nueva_evidencia_rag = evidencia_anterior + [str(evidencia)]

    total_caracteres = sum(len(str(item)) for item in nueva_evidencia_rag)

    return {
        "query_rag": query,
        "evidencia_rag": nueva_evidencia_rag,
        "caracteres_evidencia_rag": total_caracteres,
        "evidencia_rag_resumida": False,
        "herramientas_usadas": ["search_rag"],
        "intentos_rag": state["intentos_rag"] + 1,
        "iteraciones_herramientas": state["iteraciones_herramientas"] + 1,
    }


def _resumen_mock_evidencia_rag(state: EstadoClinico) -> str:
    """
    Resumen determinista para modo mock.
    Sirve para probar el flujo sin gastar API.
    """
    evidencia = state.get("evidencia_rag", [])
    texto = "\n\n".join(str(item) for item in evidencia)

    resumen = f"""
    Resumen compacto de evidencia RAG recuperada:
    - La evidencia recuperada se relaciona con trastornos de presión arterial y enfermedades cardiovasculares asociadas.
    - Se deben considerar los valores de presión arterial, MAP, síntomas, antecedentes y comorbilidades.
    - La evidencia debe usarse como apoyo al razonamiento clínico, no como diagnóstico definitivo.
    - Texto original compactado desde {len(texto)} caracteres.
    """.strip()

    return resumen


def nodo_resumir_evidencia_rag(state: EstadoClinico):
    """
    Nodo LLM resumidor de evidencia RAG.

    Objetivo:
    Reducir el tamaño de evidencia_rag cuando la recuperación desde RAG
    genera demasiado contexto para los nodos LLM posteriores.
    """

    print("[LLM Resumidor RAG] Compactando evidencia recuperada")

    evidencia = state.get("evidencia_rag", [])
    texto_evidencia = "\n\n--- EVIDENCIA ---\n\n".join(str(item) for item in evidencia)

    if not USE_REAL_LLM:
        resumen = _resumen_mock_evidencia_rag(state)
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Eres un agente resumidor de evidencia clínica recuperada por RAG.

                    Tu tarea es compactar evidencia médica para que pueda ser usada por
                    un sistema multiagente clínico sin saturar la ventana de contexto.

                    Reglas:
                    - No inventes información.
                    - No agregues diagnóstico definitivo.
                    - No agregues prescripción farmacológica automática.
                    - Conserva diagnósticos diferenciales si aparecen.
                    - Conserva datos clínicamente útiles: umbrales, definiciones,
                      criterios, signos de alarma, diagnósticos diferenciales,
                      relación con presión arterial, MAP, hipotensión, hipertensión
                      y riesgo cardiovascular.
                    - El resultado debe ser breve, estructurado y útil para el analizador.
                    """,
                ),
                (
                    "user",
                    """
                    Caso clínico:
                    {caso_clinico}

                    Datos extraídos:
                    {datos_extraidos}

                    Evidencia RAG acumulada:
                    {texto_evidencia}

                    Genera un resumen clínico compacto en máximo 8 viñetas.
                    """,
                ),
            ]
        )

        # Puedes usar llm_investigador para no declarar otro modelo.
        # Si luego quieres separar responsabilidades, crea llm_resumidor en llms.py.
        chain = prompt | llm_openai

        respuesta = chain.invoke(
            {
                "caso_clinico": state["caso_clinico"],
                "datos_extraidos": state["datos_extraidos"],
                "texto_evidencia": texto_evidencia,
            }
        )

        resumen = respuesta.content

    resumen = resumen.strip()

    return {
        # Reemplazo directo:
        # La evidencia completa se cambia por una versión compacta.
        "evidencia_rag": [resumen],
        "resumen_evidencia_rag": resumen,
        "evidencia_rag_resumida": True,
        "caracteres_evidencia_rag": len(resumen),
        "conteo_resumenes_rag": state.get("conteo_resumenes_rag", 0) + 1,
        "herramientas_usadas": ["resumir_evidencia_rag"],
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
# Nodo LLM 2: analisis preliminar
# =========================================================


def nodo_analizar(state: EstadoClinico):
    print("[LLM Analizador] Generando análisis preliminar")

    if not USE_REAL_LLM: # mock de depuración
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
                    Eres el Analizador Clínico de un sistema multiagente de apoyo clínico.

                    Tu tarea es generar un análisis preliminar útil para un médico, usando:
                    - caso clínico original,
                    - datos extraídos,
                    - MAP,
                    - predicción cardiovascular ML si existe,
                    - evidencia RAG compactada.

                    No entregues diagnóstico definitivo.
                    No prescribas tratamiento de forma automática.
                    Declara limitaciones cuando falten datos.
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
        chain = prompt | llm_openai
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
# Nodo LLM 3: Juez
# =========================================================


def _revision_mock_juez(state: EstadoClinico) -> RevisionJuez:
    if len(state["evidencia_rag"]) == 0:
        estado_revision = "falta_evidencia"
        observacion = "No existe evidencia clinica recuperada desde el RAG."
    elif "Evidencia RAG simulada por error" in str(state["evidencia_rag"]):
        estado_revision = "falta_evidencia"
        observacion = "La evidencia proviene de un fallback simulado por error del RAG."
    elif "diagnostico definitivo" in state["analisis"].lower() or "diagnostico definitivo" in state["analisis"].lower():
        estado_revision = "riesgo_alucinacion"
        observacion = "El analisis contiene lenguaje que podria interpretarse como diagnostico definitivo."
    else:
        estado_revision = "aprobado"
        observacion = "La respuesta esta suficientemente fundamentada para apoyo clinico."

    print(f"[LLM Juez] Revision: {estado_revision}")
    print(f"[LLM Juez] Observacion: {observacion}")

    return RevisionJuez(estado_revision=estado_revision, observacion=observacion)

def nodo_juez(state: EstadoClinico):
    print("[LLM Juez] Evaluando suficiencia de la informacion")

    if not USE_REAL_LLM:
        revision = _revision_mock_juez(state)
    else:
        llm_estructurado = llm_openai.with_structured_output(RevisionJuez)

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

        print(f"[LLM Juez] Revision: {revision.estado_revision}")
        print(f"[LLM Juez] Observacion: {revision.observacion}")

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
        REPORTE DE APOYO CLINICO CON LIMITACIONES

        Caso ingresado:
        {state["caso_clinico"]}

        Datos extraidos:
        {state["datos_extraidos"]}

        MAP:
        {state["map_value"]}

        Interpretacion MAP:
        {state["interpretacion_map"]}

        Analisis preliminar:
        {state["analisis"]}

        Estado de revision:
        {state["revision"]}

        Observacion del juez:
        {state["observacion_juez"]}

        Limitacion:
        (Se llego al limite de iteraciones permitidas)
        El sistema no logra una aprobacion completa del juez dentro del maximo de ciclos permitido.
        Esta salida debe interpretarse solo como apoyo preliminar y requiere revision del profesional medico.

        Nota:
        El sistema no entrega diagnostico definitivo ni prescribe tratamiento.
        """
    else:
        respuesta = f"""
        REPORTE DE APOYO CLINICO

        Caso ingresado:
        {state["caso_clinico"]}

        Datos extraidos:
        {state["datos_extraidos"]}

        MAP:
        {state["map_value"]}

        Interpretacion MAP:
        {state["interpretacion_map"]}

        Evidencia usada:
        {state["evidencia_rag"]}

        Analisis:
        {state["analisis"]}

        Nota:
        El sistema entrega apoyo al razonamiento clinico. No reemplaza el criterio profesional medico.
        """

    return {"respuesta_final": respuesta.strip()}
