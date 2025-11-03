import pandas as pd
import joblib
import os
import numpy as np

def run_gradual_prediction(filepath, models_folder, output_folder):
    """
    Realiza predicciones graduales "inteligentes".
    Detecta automáticamente qué unidad (U1, U2, U3, U4) debe predecir
    basándose en las columnas de notas presentes/ausentes en el archivo.
    """
    print("--- Iniciando Predicción Gradual Inteligente ---")

    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"Error leyendo el archivo: {e}. Intentando con CSV...")
        try:
            df = pd.read_csv(filepath)
        except Exception as e_csv:
            raise Exception(f"No se pudo leer el archivo ni como Excel ni como CSV: {e_csv}")

    # --- Lógica de Detección de Objetivo ---
    
    # Características base (siempre deben estar)
    BASE_FEATURES = [
        'EDAD', 'MODALIDAD', 'Carrera', 'CURSO', 
        'TIPOSESION', 'SECCIÓN', 'DOCENTE', 'PERIODO',
        'PORCENTAJE_asistencia', 'CUOTA_1', 'CUOTA_2', 'CUOTA_3', 'CUOTA_4', 'CUOTA_5'
    ]
    
    model_to_load = None
    preprocessor_to_load = None
    features_to_use = []
    prediction_target = None

    # Función para verificar si una columna de nota está vacía o no existe
    def is_note_empty(df, col_name):
        if col_name not in df.columns:
            return True
        # Rellenar 'NP' o strings vacíos con NaN, luego verificar si todos son nulos
        return pd.to_numeric(df[col_name], errors='coerce').isnull().all()

    # Iniciar la lógica de decisión en cadena
    if is_note_empty(df, 'U1'):
        print("Detectado: Faltan notas de U1. Prediciendo U1...")
        prediction_target = 'U1'
        features_to_use = BASE_FEATURES
        
    elif is_note_empty(df, 'U2'):
        print("Detectado: U1 presente, U2 falta. Prediciendo U2...")
        prediction_target = 'U2'
        features_to_use = BASE_FEATURES + ['U1']
        df['U1'] = pd.to_numeric(df['U1'], errors='coerce').fillna(0) # Limpiar U1 para usarla como feature

    elif is_note_empty(df, 'U3'):
        print("Detectado: U1 y U2 presentes, U3 falta. Prediciendo U3...")
        prediction_target = 'U3'
        features_to_use = BASE_FEATURES + ['U1', 'U2']
        df['U1'] = pd.to_numeric(df['U1'], errors='coerce').fillna(0)
        df['U2'] = pd.to_numeric(df['U2'], errors='coerce').fillna(0)

    elif is_note_empty(df, 'U4'):
        print("Detectado: U1, U2, U3 presentes, U4 falta. Prediciendo U4...")
        prediction_target = 'U4'
        features_to_use = BASE_FEATURES + ['U1', 'U2', 'U3']
        df['U1'] = pd.to_numeric(df['U1'], errors='coerce').fillna(0)
        df['U2'] = pd.to_numeric(df['U2'], errors='coerce').fillna(0)
        df['U3'] = pd.to_numeric(df['U3'], errors='coerce').fillna(0)

    else:
        print("Todas las notas (U1-U4) están presentes. No se requiere predicción gradual.")
        return df, "No se requiere predicción, todas las notas están completas.", "Completo"

    # --- Carga de Modelo y Preprocesador ---
    model_filename = f"modelo_gradual_{prediction_target}.pkl"
    preprocessor_filename = f"preprocessor_gradual_{prediction_target}.pkl"
    
    model_path = os.path.join(models_folder, model_filename)
    preprocessor_path = os.path.join(models_folder, preprocessor_filename)

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"No se encontraron los archivos necesarios: {model_filename} o {preprocessor_filename}. "
                              f"Asegúrese de entrenar el modelo para {prediction_target} primero.")

    print(f"Cargando modelo: {model_path}")
    model = joblib.load(model_path)
    print(f"Cargando preprocesador: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)

    # --- Preparación de Datos y Predicción ---
    
    # Asegurarse de que el DF de predicción tenga las columnas que el preprocesador espera
    X_pred = df.copy()
    
    # Añadir columnas faltantes con NaN si no existen en el archivo de predicción
    # (Esto es vital si el preprocesador fue entrenado con más columnas)
    expected_features = preprocessor.feature_names_in_
    for col in expected_features:
        if col not in X_pred.columns:
            X_pred[col] = np.nan
            
    # Reordenar columnas para que coincidan exactamente con el entrenamiento
    X_pred = X_pred[expected_features]

    print(f"Ejecutando preprocesamiento en {len(X_pred)} filas...")
    X_processed = preprocessor.transform(X_pred)

    print("Realizando predicciones...")
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)[:, 1] # Probabilidad de 'Aprobado' (clase 1)

    # --- Formateo de Resultados ---
    df_results = df[['CÓDIGO', 'ALUMNO', 'Carrera', 'CURSO']].copy()
    df_results[f'Prediccion_{prediction_target}'] = np.where(predictions == 1, 'Aprobado', 'Desaprobado')
    df_results[f'Prob_Aprobar_{prediction_target}'] = (probabilities * 100).round(2)

    # Guardar resultados
    results_filename = f"resultados_prediccion_{prediction_target}.csv"
    results_path = os.path.join(output_folder, results_filename)
    df_results.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"Resultados guardados en: {results_path}")

    return df_results, results_filename, prediction_target
