import pandas as pd
import joblib
import os
import numpy as np

# Mapeo de nombres de columnas
COLUMN_MAP = {
    'DESCRIPCION': 'Carrera',
    'DESCRIPCION_1': 'CURSO' 
}

# Definición de Características Base (debe coincidir con el entrenamiento)
BASE_FEATURES = [
    'EDAD', 'MODALIDAD', 'Carrera', 'CURSO', 
    'TIPOSESION', 'SECCIÓN', 'DOCENTE', 'PERIODO',
    'PORCENTAJE_asistencia', 
    'CUOTA_1', 'CUOTA_2', 'CUOTA_3', 'CUOTA_4', 'CUOTA_5'
]

def run_gradual_prediction(filepath, models_folder, output_folder):
    """
    Ejecuta el Proceso 2 (Predicción):
    Detecta qué unidad predecir (U1, U2, U3, U4) basándose en las
    columnas de notas presentes en el archivo subido.
    """
    print(f"--- Iniciando Predicción Gradual ---")
    
    try:
        df = pd.read_excel(filepath)
    except Exception:
        df = pd.read_csv(filepath)
        
    df.rename(columns=COLUMN_MAP, inplace=True)
    df_original = df.copy() # Guardar original para el reporte final

    # --- Lógica de Detección Automática ---
    target_unit = None
    features_to_use = []
    
    # Convertir notas a numérico para chequear nulos
    for col in ['U1', 'U2', 'U3', 'U4']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan # Asegurarse de que la columna exista para la lógica

    # Chequear en orden qué predecir
    if df['U1'].isnull().all():
        print("Detectado: Faltan notas U1. Prediciendo U1.")
        target_unit = 'U1'
        features_to_use = BASE_FEATURES
    elif df['U2'].isnull().all():
        print("Detectado: U1 presente, U2 falta. Prediciendo U2.")
        target_unit = 'U2'
        features_to_use = BASE_FEATURES + ['U1']
    elif df['U3'].isnull().all():
        print("Detectado: U1/U2 presentes, U3 falta. Prediciendo U3.")
        target_unit = 'U3'
        features_to_use = BASE_FEATURES + ['U1', 'U2']
    elif df['U4'].isnull().all():
        print("Detectado: U1/U2/U3 presentes, U4 falta. Prediciendo U4.")
        target_unit = 'U4'
        features_to_use = BASE_FEATURES + ['U1', 'U2', 'U3']
    else:
        print("Detectado: Todas las notas (U1-U4) están presentes. No hay nada que predecir.")
        return df_original, "N/A", "Completo"

    model_filename = f"modelo_gradual_{target_unit}.pkl"
    preprocessor_filename = f"preprocessor_gradual_{target_unit}.pkl"
    
    model_path = os.path.join(models_folder, model_filename)
    preprocessor_path = os.path.join(models_folder, preprocessor_filename)

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"No se encontraron los archivos necesarios: {model_filename} o {preprocessor_filename}. "
                            f"Por favor, entrene este modelo ('{target_unit}') primero usando la data consolidada.")

    # --- Cargar Modelo y Preprocesador ---
    print(f"Cargando modelo: {model_path}")
    model = joblib.load(model_path)
    print(f"Cargando preprocesador: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)

    # Asegurarse de que todas las columnas existan en el DF de predicción
    final_features_list = []
    for col in features_to_use:
        if col not in df.columns:
            print(f"Advertencia: Falta la columna '{col}' en el archivo de predicción. Rellenando con 0/NaN.")
            if col in ['U1', 'U2', 'U3', 'EDAD', 'PORCENTAJE_asistencia']:
                df[col] = 0
            else:
                 df[col] = np.nan # Dejar que el imputer se encargue
        final_features_list.append(col)

    # Limpiar notas (U1, U2, U3) usadas como CARACTERÍSTICAS
    for col in ['U1', 'U2', 'U3']:
        if col in final_features_list:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    X_predict = df[final_features_list]

    # --- Preprocesar y Predecir ---
    print("Aplicando preprocesador a nuevos datos...")
    X_processed = preprocessor.transform(X_predict)

    print("Realizando predicciones...")
    predicciones = model.predict(X_processed)
    probabilidades = model.predict_proba(X_processed)[:, 1] # Probabilidad de '1' (Aprobado)

    # --- Formatear Resultados ---
    df_original[f'Prediccion_{target_unit}'] = ["Aprobado" if p == 1 else "Desaprobado" for p in predicciones]
    df_original[f'Prob_Aprobar_{target_unit}'] = [f"{prob * 100:.2f}%" for prob in probabilidades]
    
    # Guardar resultados
    results_filename = f"resultados_prediccion_{target_unit}.csv"
    results_filepath = os.path.join(output_folder, results_filename)
    df_original.to_csv(results_filepath, index=False)
    print(f"Resultados guardados en: {results_filepath}")

    return df_original, results_filename, target_unit
