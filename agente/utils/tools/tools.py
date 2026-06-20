from langchain_core.tools import tool
from typing import Optional
import pandas as pd
import json

from .search import search, format_search_results
from clasificador_Util.clasificador_paciente_cardiovascular import FEATURES, _crear_dataframe_entrada, inicializar_modelo_cardiovascular


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
def clasificar_paciente_cardiovascular(
    age: Optional[float] = None,
    sex: Optional[float] = None,
    cp: Optional[float] = None,
    trestbps: Optional[float] = None,
    chol: Optional[float] = None,
    fbs: Optional[float] = None,
    restecg: Optional[float] = None,
    thalach: Optional[float] = None,
    exang: Optional[float] = None,
    oldpeak: Optional[float] = None,
    slope: Optional[float] = None,
    ca: Optional[float] = None,
    thal: Optional[float] = None,
) -> str:
    """
    Clasifica si el paciente presenta o no una posible enfermedad cardiovascular.

    La clasificación es realizada por un modelo de ML XGBoost preentrenado con el
    dataset binario Heart Disease Dataset de Kaggle:
    https://www.kaggle.com/datasets/yasserh/heart-disease-dataset/data

    Parámetros esperados según el dataset:
    - age: edad del paciente.
    - sex: sexo codificado. Usualmente 1 = masculino, 0 = femenino.
    - cp: tipo de dolor torácico codificado.
    - trestbps: presión arterial en reposo.
    - chol: colesterol sérico.
    - fbs: glucosa en ayunas > 120 mg/dl. Usualmente 1 = sí, 0 = no.
    - restecg: resultado electrocardiográfico en reposo codificado.
    - thalach: frecuencia cardíaca máxima alcanzada.
    - exang: angina inducida por ejercicio. Usualmente 1 = sí, 0 = no.
    - oldpeak: depresión ST inducida por ejercicio respecto al reposo.
    - slope: pendiente del segmento ST en ejercicio.
    - ca: número de vasos principales observados por fluoroscopía.
    - thal: talasemia codificada.

    Retorna:
    String JSON con predicción, probabilidad, variables faltantes y advertencias.
    """

    entrada = _crear_dataframe_entrada(
        age=age,
        sex=sex,
        cp=cp,
        trestbps=trestbps,
        chol=chol,
        fbs=fbs,
        restecg=restecg,
        thalach=thalach,
        exang=exang,
        oldpeak=oldpeak,
        slope=slope,
        ca=ca,
        thal=thal,
    )

    variables_faltantes = [
        col for col in FEATURES if pd.isna(entrada.loc[0, col])
    ]
    variables_recibidas = [
        col for col in FEATURES if col not in variables_faltantes
    ]

    # Control de seguridad: XGBoost + imputer puede predecir con pocos datos,
    # pero clínicamente no conviene emitir una predicción si casi todo falta.
    if len(variables_recibidas) < 5:
        resultado = {
            "herramienta": "clasificar_paciente_cardiovascular",
            "estado": "prediccion_no_realizada",
            "motivo": "Datos insuficientes para una predicción mínimamente informativa.",
            "minimo_requerido": "Al menos 5 variables del dataset.",
            "variables_recibidas": variables_recibidas,
            "variables_faltantes": variables_faltantes,
            "advertencia_clinica": (
                "Esta herramienta no reemplaza el criterio médico. "
                "Debe usarse solo como apoyo al análisis del sistema multiagente."
            ),
        }
        return json.dumps(resultado, ensure_ascii=False, indent=2)

    try:
        modelo = inicializar_modelo_cardiovascular()
        prediccion = int(modelo.predict(entrada)[0])

        if hasattr(modelo, "predict_proba"):
            probabilidad = float(modelo.predict_proba(entrada)[0][1])
        else:
            probabilidad = None

        clasificacion = (
            "posible_presencia_enfermedad_cardiovascular"
            if prediccion == 1
            else "sin_indicador_ml_de_enfermedad_cardiovascular"
        )

        advertencias = [
            "Predicción generada por modelo ML; no constituye diagnóstico clínico.",
            "Interpretar junto con síntomas, examen físico, ECG, laboratorio y evidencia RAG.",
        ]

        if variables_faltantes:
            advertencias.append(
                "La predicción fue calculada con variables faltantes imputadas por el pipeline."
            )

        resultado = {
            "herramienta": "clasificar_paciente_cardiovascular",
            "estado": "prediccion_realizada",
            "modelo": "XGBoostClassifier",
            "dataset_base": "Kaggle - Heart Disease Dataset / UCI-derived binary target",
            "prediccion_binaria": prediccion,
            "clasificacion": clasificacion,
            "probabilidad_enfermedad_cardiovascular": probabilidad,
            "umbral_decision": 0.5,
            "variables_recibidas": variables_recibidas,
            "variables_faltantes": variables_faltantes,
            "advertencias": advertencias,
        }

        return json.dumps(resultado, ensure_ascii=False, indent=2)

    except Exception as exc:
        resultado = {
            "herramienta": "clasificar_paciente_cardiovascular",
            "estado": "error_modelo",
            "motivo": str(exc),
            "variables_recibidas": variables_recibidas,
            "variables_faltantes": variables_faltantes,
        }
        return json.dumps(resultado, ensure_ascii=False, indent=2)