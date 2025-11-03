import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename

# Importamos las funciones actualizadas que nos ayudarán
from pipelines.preprocessing_gradual import (
    clean_data,
    ensure_expected_features,
    feature_engineering_by_stage,
    standardize_gradual_columns,
)

def run_gradual_prediction(data_path, model_path):
    """
    Realiza predicciones inteligentes, alineando perfectamente los datos de predicción con los de entrenamiento.
    """
    stage = "desconocida"
    try:
        # 1. Cargar modelo y datos
        model_pipeline = joblib.load(model_path)
        df_raw = pd.read_excel(data_path)
        df_original = df_raw.copy()
        df_new = standardize_gradual_columns(df_raw)

        # 2. DETECCIÓN DE ETAPA (del nombre del modelo)
        if "ud1" in model_path.lower(): stage = 1
        elif "ud2" in model_path.lower(): stage = 2
        elif "ud3" in model_path.lower(): stage = 3
        elif "ud4" in model_path.lower(): stage = 4
        else:
            raise ValueError("No se pudo determinar la etapa desde el nombre del archivo del modelo.")
        print(f"--- Modelo de Etapa {stage} detectado. Preparando datos... ---")

        # 3. LIMPIEZA DE DATOS (Convertir 'COBRADO' a 1, etc.)
        df_cleaned = clean_data(df_new.copy())

        # 5. CREACIÓN DE CARACTERÍSTICAS
        df_prepared = feature_engineering_by_stage(df_cleaned.copy(), stage)
        
        # 6. ALINEACIÓN FINAL Y ROBUSTA
        df_aligned = ensure_expected_features(df_prepared, stage)

        # 7. PREDICCIÓN
        predictions = model_pipeline.predict(df_aligned)
        probabilities = model_pipeline.predict_proba(df_aligned)[:, 1]

        df_original['PREDICCION'] = ['Aprobado' if p == 1 else 'Desaprobado' for p in predictions]
        df_original['PROBABILIDAD_APROBAR'] = [f"{p:.2%}" for p in probabilities]
        
        # 8. Guardar resultados
        base_name = os.path.basename(data_path)
        file_name, _ = os.path.splitext(base_name)
        output_filename = f"resultados_{file_name}_con_modelo_UD{stage}.xlsx"
        output_path = os.path.join("uploads", "prediction", secure_filename(output_filename))
        df_original.to_excel(output_path, index=False)
        
        return df_original, output_path, None

    except Exception as e:
        error_message = f"Error durante la predicción: {e}. Revisa las columnas del archivo."
        print(error_message)
        return None, None, error_message