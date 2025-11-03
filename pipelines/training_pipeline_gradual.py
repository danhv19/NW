import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from .preprocessing_gradual import preprocess_gradual
# from .plotting_gradual import plot_gradual_results # <--- LÍNEA COMENTADA

def run_gradual_training(filepath, target_unit, models_folder, plots_folder):
    """
    Entrena modelos para una unidad específica (U1, U2, U3, o U4).
    
    Args:
        filepath (str): Ruta al archivo de datos de entrenamiento.
        target_unit (str): La unidad a predecir ('U1', 'U2', 'U3', 'U4').
        models_folder (str): Carpeta donde se guardarán los modelos.
        plots_folder (str): Carpeta donde se guardarán los gráficos.
    """
    print(f"--- Iniciando Entrenamiento Gradual para: {target_unit} ---")

    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"Error leyendo el archivo: {e}. Intentando con CSV...")
        try:
            df = pd.read_csv(filepath)
        except Exception as e_csv:
            raise Exception(f"No se pudo leer el archivo ni como Excel ni como CSV: {e_csv}")

    # --- Definición de Características y Objetivo ---
    # Características base (datos de matrícula)
    BASE_FEATURES = [
        'EDAD', 'MODALIDAD', 'Carrera', 'CURSO', 
        'TIPOSESION', 'SECCIÓN', 'DOCENTE', 'PERIODO',
        'PORCENTAJE_asistencia', 'CUOTA_1', 'CUOTA_2', 'CUOTA_3', 'CUOTA_4', 'CUOTA_5'
    ]
    
    # Diccionario para manejar la lógica gradual
    training_map = {
        'U1': {
            'target_col': 'U1',
            'features': BASE_FEATURES
        },
        'U2': {
            'target_col': 'U2',
            'features': BASE_FEATURES + ['U1']
        },
        'U3': {
            'target_col': 'U3',
            'features': BASE_FEATURES + ['U1', 'U2']
        },
        'U4': {
            'target_col': 'U4',
            'features': BASE_FEATURES + ['U1', 'U2', 'U3']
        }
    }

    if target_unit not in training_map:
        raise ValueError(f"Unidad objetivo '{target_unit}' no es válida. Debe ser 'U1', 'U2', 'U3', o 'U4'.")

    config = training_map[target_unit]
    TARGET_COLUMN = config['target_col']
    FEATURES = [f for f in config['features'] if f in df.columns]

    print(f"Objetivo: {TARGET_COLUMN}")
    print(f"Características usadas: {FEATURES}")

    # --- Preprocesamiento ---
    # Crear la variable objetivo binaria (Aprobado/Desaprobado)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    df.dropna(subset=[TARGET_COLUMN], inplace=True) # Eliminar filas donde el objetivo es nulo
    df[f'{TARGET_COLUMN}_Aprobado'] = (df[TARGET_COLUMN] > 10.5).astype(int)
    
    TARGET = f'{TARGET_COLUMN}_Aprobado'
    
    # Asegurarse de que las características de notas (si se usan) sean numéricas
    for col in ['U1', 'U2', 'U3']:
        if col in FEATURES:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    X = df[FEATURES]
    y = df[TARGET]

    # Preprocesar los datos
    X_processed, preprocessor = preprocess_gradual(X)
    
    # Guardar el preprocesador
    preprocessor_filename = f"preprocessor_gradual_{target_unit}.pkl"
    joblib.dump(preprocessor, os.path.join(models_folder, preprocessor_filename))
    print(f"Preprocesador guardado en: {preprocessor_filename}")

    # --- División de Datos ---
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

    # --- Entrenamiento de 3 Modelos ---
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }

    metrics = {}
    best_model = None
    best_accuracy = 0.0

    print("\n--- Entrenando Modelos ---")
    for name, model in models.items():
        print(f"Entrenando {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics[name] = {
            'accuracy': accuracy,
            'precision_aprobado': report.get('1', {}).get('precision', 0),
            'recall_aprobado': report.get('1', {}).get('recall', 0),
            'f1_aprobado': report.get('1', {}).get('f1-score', 0)
        }
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    # Guardar el mejor modelo
    model_filename = f"modelo_gradual_{target_unit}.pkl"
    joblib.dump(best_model, os.path.join(models_folder, model_filename))
    print(f"Mejor modelo ({type(best_model).__name__}) guardado como: {model_filename}")

    # --- Generación de Gráficos ---
    # plots = plot_gradual_results(df, target_unit, plots_folder) # <--- LÍNEA COMENTADA
    plots = {} # <--- LÍNEA AÑADIDA para que 'plots' exista

    return metrics, plots, model_filename

