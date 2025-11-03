import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from .preprocessing_gradual import preprocess_gradual # <--- Usa el preprocesador gradual

# --- Definición de Características Unificadas ---
# Estas son las características de MATRÍCULA
BASE_FEATURES = [
    'EDAD', 'MODALIDAD', 'Carrera', 'CURSO', 
    'TIPOSESION', 'SECCIÓN', 'DOCENTE', 'PERIODO',
    'PORCENTAJE_asistencia', 
    'CUOTA_1', 'CUOTA_2', 'CUOTA_3', 'CUOTA_4', 'CUOTA_5'
]

# Mapeo de nombres de columnas
COLUMN_MAP = {
    'DESCRIPCION': 'Carrera',
    'DESCRIPCION_1': 'CURSO' 
}

def run_gradual_training(filepath, target_unit, models_folder):
    """
    Ejecuta el Proceso 2 (Entrenamiento):
    Entrena un modelo para una unidad específica (U1, U2, U3, o U4).
    Usa el "Data - consolidada.xlsx" (que debe tener U1-U4) para crear
    los modelos pkl graduales.
    """
    print(f"--- Iniciando Entrenamiento Gradual para: {target_unit} ---")

    try:
        df = pd.read_excel(filepath)
    except Exception:
        df = pd.read_csv(filepath)

    df.rename(columns=COLUMN_MAP, inplace=True)

    # --- Definición de Características y Objetivo ---
    # El mapa define qué características usar para entrenar CADA modelo
    training_map = {
        'U1': {
            'target_col': 'U1',
            'features': BASE_FEATURES # Modelo U1 solo usa datos de matrícula
        },
        'U2': {
            'target_col': 'U2',
            'features': BASE_FEATURES + ['U1'] # Modelo U2 usa matrícula + U1
        },
        'U3': {
            'target_col': 'U3',
            'features': BASE_FEATURES + ['U1', 'U2'] # Modelo U3 usa matrícula + U1 + U2
        },
        'U4': {
            'target_col': 'U4',
            'features': BASE_FEATURES + ['U1', 'U2', 'U3'] # Modelo U4 usa matrícula + U1 + U2 + U3
        }
    }

    if target_unit not in training_map:
        raise ValueError(f"Unidad objetivo '{target_unit}' no es válida. Debe ser 'U1', 'U2', 'U3', o 'U4'.")

    config = training_map[target_unit]
    TARGET_COLUMN_NAME = config['target_col']
    
    # Asegurarse de que el archivo de entrenamiento tenga todas las columnas necesarias
    if TARGET_COLUMN_NAME not in df.columns:
        # Intentar con PROMEDIO_CURSO si U1-U4 no están (caso especial para data consolidada)
        if 'PROMEDIO_CURSO' in df.columns and target_unit == 'U1':
            print(f"Advertencia: No se encontró '{TARGET_COLUMN_NAME}', usando 'PROMEDIO_CURSO' para entrenar el modelo U1.")
            TARGET_COLUMN_NAME = 'PROMEDIO_CURSO'
        else:
            raise ValueError(f"El archivo de entrenamiento no tiene la columna objetivo '{TARGET_COLUMN_NAME}'.")
    
    # Verificar todas las características
    available_features = []
    for f in config['features']:
        if f in df.columns:
            available_features.append(f)
        elif f in ['U1', 'U2', 'U3'] and f not in df.columns:
             raise ValueError(f"Para entrenar {target_unit}, el archivo debe tener la columna '{f}'.")
        # Ignorar si falta una característica base (ej. PROCEDENCIA)

    FEATURES = available_features
    print(f"Objetivo: {TARGET_COLUMN_NAME}")
    print(f"Características usadas: {FEATURES}")

    # --- Preprocesamiento del Objetivo y Características ---
    df[TARGET_COLUMN_NAME] = pd.to_numeric(df[TARGET_COLUMN_NAME], errors='coerce')
    df.dropna(subset=[TARGET_COLUMN_NAME], inplace=True) 
    
    # Convertir a 0 (Desaprobado) y 1 (Aprobado)
    df[f'TARGET_Aprobado'] = (df[TARGET_COLUMN_NAME] > 10.5).astype(int)
    TARGET = 'TARGET_Aprobado'
    
    # Limpiar columnas de notas usadas como características
    for col in ['U1', 'U2', 'U3']:
        if col in FEATURES:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Rellenar notas faltantes con 0

    X = df[FEATURES]
    y = df[TARGET]

    # Usar el preprocesador gradual
    X_processed, preprocessor = preprocess_gradual(X) # <--- Ya no necesita target_unit
    
    # Guardar el preprocesador específico para esta unidad
    preprocessor_filename = f"preprocessor_gradual_{target_unit}.pkl"
    joblib.dump(preprocessor, os.path.join(models_folder, preprocessor_filename))
    print(f"Preprocesador guardado en: {preprocessor_filename}")

    # --- División de Datos ---
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

    # --- Entrenamiento de 3 Modelos ---
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100),
        'GradientBoosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
    }

    metrics = {}
    best_model = None
    best_accuracy = 0.0

    print("\n--- Entrenando 3 Modelos ---")
    for name, model in models.items():
        print(f"Entrenando {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        metrics[name] = {
            'accuracy': accuracy,
            'precision_aprobado': report.get('1', {}).get('precision', 0),
            'recall_aprobado': report.get('1', {}).get('recall', 0),
            'f1_aprobado': report.get('1', {}).get('f1-score', 0)
        }
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            metrics['best_model'] = name # Guardar nombre del mejor modelo

    # Guardar el mejor modelo
    model_filename = f"modelo_gradual_{target_unit}.pkl"
    joblib.dump(best_model, os.path.join(models_folder, model_filename))
    print(f"Mejor modelo ({type(best_model).__name__}) guardado como: {model_filename}")
    
    return metrics, model_filename
