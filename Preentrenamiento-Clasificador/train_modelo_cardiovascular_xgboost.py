"""
Entrenamiento del modelo XGBoost para la herramienta clasificar_paciente_cardiovascular.

Dataset recomendado:
    https://www.kaggle.com/datasets/yasserh/heart-disease-dataset/data

Uso:
    python train_modelo_cardiovascular_xgboost.py --csv ruta/al/heart.csv

Salida:
    modelo_ml/xgboost_heart_pipeline.joblib
    modelo_ml/xgboost_heart_metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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

CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "modelo_ml"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "xgboost_heart_pipeline.joblib"
DEFAULT_METADATA_PATH = DEFAULT_MODEL_DIR / "xgboost_heart_metadata.json"


def _normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas para evitar errores por mayúsculas/espacios."""
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def _obtener_target(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """
    Obtiene el target binario.

    Para el dataset YasserH se espera la columna `target`.
    Para variantes UCI con columna `num`, transforma:
        0 -> 0
        1,2,3,4 -> 1
    """
    if "target" in df.columns:
        y = pd.to_numeric(df["target"], errors="coerce")
        target_name = "target"
    elif "num" in df.columns:
        y = pd.to_numeric(df["num"], errors="coerce")
        y = (y > 0).astype("float")
        target_name = "num_binarizado"
    else:
        raise ValueError("No se encontró columna objetivo. Se esperaba `target` o `num`.")

    return y, target_name


def preparar_dataset(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Carga y limpia el dataset.

    Limpieza aplicada:
    1. Normalización de nombres de columnas.
    2. Eliminación de duplicados.
    3. Validación de columnas esperadas.
    4. Conversión de variables a formato numérico.
    5. Conversión del target a clasificación binaria.
    6. Eliminación de filas sin target.
    """
    df = pd.read_csv(csv_path)
    df = _normalizar_columnas(df)

    # El dataset de YasserH suele tener columnas ya limpias y numéricas.
    # Mantener este paso ayuda si el CSV fue editado manualmente.
    df = df.drop_duplicates().reset_index(drop=True)

    missing_features = [col for col in FEATURES if col not in df.columns]
    if missing_features:
        raise ValueError(
            "Faltan columnas requeridas por el modelo: "
            + ", ".join(missing_features)
        )

    y, target_name = _obtener_target(df)

    X = df[FEATURES].copy()
    for col in FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    mask_valid_target = y.notna()
    X = X.loc[mask_valid_target].reset_index(drop=True)
    y = y.loc[mask_valid_target].astype(int).reset_index(drop=True)

    unique_targets = sorted(y.unique().tolist())
    if unique_targets not in ([0, 1], [0], [1]):
        raise ValueError(f"Target no binario detectado: {unique_targets}")

    return X, y, target_name


def construir_pipeline_xgboost() -> Pipeline:
    """
    Construye el pipeline serializable:
    - Imputación mediana para variables continuas.
    - Imputación de moda + OneHotEncoder para variables categóricas codificadas.
    - Clasificador XGBoost.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", numeric_transformer, CONTINUOUS_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )


def entrenar_y_guardar_modelo(
    csv_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> dict:
    """
    Entrena el pipeline XGBoost y lo guarda como archivo .joblib.
    Retorna métricas de evaluación para documentar el entrenamiento.
    """
    X, y, target_name = preparar_dataset(csv_path)

    if len(y.unique()) < 2:
        raise ValueError("El dataset necesita tener ambas clases: 0 y 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = construir_pipeline_xgboost()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            zero_division=0,
            output_dict=True,
        ),
    }

    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)

    metadata = {
        "model_type": "XGBClassifier",
        "problem_type": "binary_classification",
        "dataset_source": "Kaggle - Heart Disease Dataset / UCI-derived",
        "target_column_used": target_name,
        "features": FEATURES,
        "continuous_features": CONTINUOUS_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "test_size": 0.20,
        "random_state": 42,
        "metrics": metrics,
    }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        required=True,
        help="Ruta al CSV del dataset heart.csv.",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Ruta donde se guardará el pipeline .joblib.",
    )
    parser.add_argument(
        "--metadata-path",
        default=str(DEFAULT_METADATA_PATH),
        help="Ruta donde se guardará la metadata JSON.",
    )

    args = parser.parse_args()

    metadata = entrenar_y_guardar_modelo(
        csv_path=args.csv,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
    )

    print("Modelo entrenado y guardado correctamente.")
    print(json.dumps(metadata["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
