"""
ToolNode: clasificador cardiovascular con XGBoost.

Entrenamiento esperado:
    python train_modelo_cardiovascular_xgboost.py --csv heart.csv

Uso en LangGraph/LangChain:
    from clasificador_paciente_cardiovascular import clasificar_paciente_cardiovascular
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Any

import joblib
import numpy as np
import pandas as pd


FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
MODEL_PATH = Path(
    os.getenv(
        "CARDIO_XGB_MODEL_PATH",
        Path(__file__).resolve().parents[4]
        / "Preentrenamiento-Clasificador"
        / "modelo_ml"
        / "xgboost_heart_pipeline.joblib",
    )
)

_model_cache = None


def inicializar_modelo_cardiovascular():
    """
    Carga el pipeline preentrenado desde disco y lo mantiene en caché.
    Esta función se puede llamar al iniciar el sistema para evitar cargar el modelo
    durante la primera invocación de la herramienta.
    """
    global _model_cache

    if _model_cache is not None:
        return _model_cache

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo XGBoost en: {MODEL_PATH}. "
            "Ejecuta primero train_modelo_cardiovascular_xgboost.py para generar el archivo .joblib."
        )

    _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def _to_float_or_nan(value: Optional[Any]) -> float:
    """Convierte valores None, '', 'null' o no numéricos en np.nan."""
    if value is None:
        return np.nan

    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"none", "null", "nan", "no informado", "desconocido"}:
            return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _crear_dataframe_entrada(
    age: Optional[float],
    sex: Optional[float],
    cp: Optional[float],
    trestbps: Optional[float],
    chol: Optional[float],
    fbs: Optional[float],
    restecg: Optional[float],
    thalach: Optional[float],
    exang: Optional[float],
    oldpeak: Optional[float],
    slope: Optional[float],
    ca: Optional[float],
    thal: Optional[float],
) -> pd.DataFrame:
    row = {
        "age": _to_float_or_nan(age),
        "sex": _to_float_or_nan(sex),
        "cp": _to_float_or_nan(cp),
        "trestbps": _to_float_or_nan(trestbps),
        "chol": _to_float_or_nan(chol),
        "fbs": _to_float_or_nan(fbs),
        "restecg": _to_float_or_nan(restecg),
        "thalach": _to_float_or_nan(thalach),
        "exang": _to_float_or_nan(exang),
        "oldpeak": _to_float_or_nan(oldpeak),
        "slope": _to_float_or_nan(slope),
        "ca": _to_float_or_nan(ca),
        "thal": _to_float_or_nan(thal),
    }

    return pd.DataFrame([row], columns=FEATURES)
