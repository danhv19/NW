import os
import joblib
import pandas as pd
import numpy as np
from pipelines.preprocessing import load_and_clean_data


def _ensure_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea CUOTAS_PENDIENTES, CUOTAS_PAGADAS, INDICE_PAGO y ASISTENCIA_POR_CURSO
    si no existen en df.
    """
    # Generar estadísticas de cuotas
    cuota_cols = [c for c in df.columns if c.startswith("CUOTA_")]
    if cuota_cols and "CUOTAS_PENDIENTES" not in df.columns:
        df["CUOTAS_PENDIENTES"] = df[cuota_cols].isnull().sum(axis=1)
    if "CUOTAS_PENDIENTES" in df.columns and "CUOTAS_PAGADAS" not in df.columns:
        df["CUOTAS_PAGADAS"] = 5 - df["CUOTAS_PENDIENTES"]
        df["INDICE_PAGO"]    = df["CUOTAS_PAGADAS"] / 5

    # Generar asistencia por curso
    if (
        "PORCENTAJE_asistencia" in df.columns
        and "CURSOS_MATRICULADOS" in df.columns
        and "ASISTENCIA_POR_CURSO" not in df.columns
    ):
        df["ASISTENCIA_POR_CURSO"] = (
            df["PORCENTAJE_asistencia"] / df["CURSOS_MATRICULADOS"]
        )

    return df


def run_prediction(data_path: str, model_path: str):
    """
    Ejecuta la fase 3 de predicción:
    - data_path: ruta al .xlsx con NUEVOS datos (sin PROMEDIO_CURSO)
    - model_path: ruta al .pkl entrenado

    Retorna:
    - df_out: DataFrame con ALUMNO, PORCENTAJE_asistencia, PREDICCION_ESTADO, CONFIANZA_APROBACION (%)
    - export_path: ruta al Excel de resultados para descarga
    """
    # 1) Cargar y limpiar datos de entrada
    df_new = load_and_clean_data(data_path, is_training_data=False)
    df_new = _ensure_derived_features(df_new)

    # 2) Cargar pipeline entrenado
    pipeline = joblib.load(model_path)

    # 3) Obtener features que espera el preprocesador
    pre = pipeline.named_steps["preprocessor"]
    try:
        feats = list(pre.feature_names_in_)
    except AttributeError:
        feats = []
        for _, _, cols in pre.transformers_:
            feats += list(cols)

    # 4) Excluir posibles columnas objetivo que aparezcan
    for drop in ("ESTADO_APROBACION", "ESTADO_APROBACION_NUM"):
        if drop in feats:
            feats.remove(drop)

    # 5) Asegurar que todas las columnas existan
    for c in feats:
        if c not in df_new.columns:
            df_new[c] = np.nan

    # 6) Construir matriz de predicción
    X_new = df_new[feats].copy()
    for c in X_new.select_dtypes(exclude="number"):
        X_new[c] = X_new[c].astype(str)
    for c in X_new.select_dtypes(include="number"):
        X_new[c] = pd.to_numeric(X_new[c], errors="coerce")

    # 7) Realizar predicción
    y_pred  = pipeline.predict(X_new)
    y_prob  = pipeline.predict_proba(X_new)[:, 1]

    # 8) Añadir resultados al DataFrame
    df_new["PREDICCION_ESTADO"]        = np.where(y_pred == 1, "Aprobado", "Desaprobado")
    df_new["CONFIANZA_APROBACION (%)"] = (y_prob * 100).round(2)

    # 9) Seleccionar sólo las columnas requeridas
    cols_out = [
        "ALUMNO",
        "PORCENTAJE_asistencia",
        "PREDICCION_ESTADO",
        "CONFIANZA_APROBACION (%)"
    ]
    df_out = df_new[[c for c in cols_out if c in df_new.columns]]

    # 10) Exportar resultados a Excel
    dest = os.path.join("uploads", "prediction")
    os.makedirs(dest, exist_ok=True)
    export_path = os.path.join(dest, "resultados_prediccion.xlsx")
    df_out.to_excel(export_path, index=False)

    return df_out, export_path
