import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from .preprocessing import preprocess_historical
from .plotting import plot_historical_results

# --- Definición de Características ---
# Columnas que esperamos del archivo "Data - consolidada.xlsx"
FEATURES = [
    'EDAD', 'MODALIDAD', 'PROCEDENCIA', 'PERIODO_INGRESO', 'DESCRIPCION', # 'DESCRIPCION' es Carrera
    'CURSOS_MATRICULADOS', 'DESCRIPCION_1', # 'DESCRIPCION_1' es CURSO
    'TIPOSESION', 'SECCIÓN', 'DOCENTE', 
    'PORCENTAJE_asistencia', 
    'CUOTA_1', 'CUOTA_2', 'CUOTA_3', 'CUOTA_4', 'CUOTA_5'
]

# Objetivo a predecir
TARGET_COLUMN = 'PROMEDIO_CURSO'

# Mapeo de nombres de columnas (si es necesario)
COLUMN_MAP = {
    'DESCRIPCION': 'Carrera',
    'DESCRIPCION_1': 'CURSO'
}

def run_historical_analysis(filepath, models_folder, plots_folder):
    """
    Ejecuta el Proceso 1: Análisis Histórico Profundo.
    Analiza el PROMEDIO_CURSO final.
    """
    print(f"--- Iniciando Análisis Histórico ---")

    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"No se pudo leer como Excel ({e}), intentando con CSV...")
        df = pd.read_csv(filepath)

    df.rename(columns=COLUMN_MAP, inplace=True)

    # --- Preprocesamiento del Objetivo ---
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{TARGET_COLUMN}' en el archivo.")
    
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    df.dropna(subset=[TARGET_COLUMN], inplace=True) 
    
    # Convertir a 0 (Desaprobado) y 1 (Aprobado)
    df['TARGET_Aprobado'] = (df[TARGET_COLUMN] > 10.5).astype(int)
    TARGET = 'TARGET_Aprobado'
    
    # Filtrar solo las características que existen en el archivo
    available_features = [f for f in FEATURES if f in df.columns]
    print(f"Características usadas: {available_features}")

    X = df[available_features]
    y = df[TARGET]

    X_processed, preprocessor = preprocess_historical(X)
    
    # Guardar el preprocesador
    preprocessor_filename = "preprocessor_historico.pkl"
    joblib.dump(preprocessor, os.path.join(models_folder, preprocessor_filename))
    print(f"Preprocesador histórico guardado en: {preprocessor_filename}")

    # --- División de Datos ---
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

    # --- Entrenamiento de 3 Modelos ---
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=150, max_depth=10),
        'GradientBoosting': GradientBoostingClassifier(random_state=42, n_estimators=150, max_depth=5, learning_rate=0.05)
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
    model_filename = "modelo_historico.pkl"
    joblib.dump(best_model, os.path.join(models_folder, model_filename))
    print(f"Mejor modelo ({type(best_model).__name__}) guardado como: {model_filename}")

    # --- Generación de Gráficos (EDA) ---
    print("Generando gráficos de análisis...")
    # Pasamos el DataFrame original (df) para el EDA
    plots = plot_historical_results(df, TARGET, available_features, plots_folder) 
    
    return metrics, plots, model_filename
