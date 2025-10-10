import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Importamos las funciones de preprocesamiento Y AHORA DE GRÁFICOS
from pipelines.preprocessing_gradual import preprocess_for_gradual_training, clean_data
from pipelines.plotting_gradual import generate_behavioral_plots # <-- AÑADIR ESTA LÍNEA

def train_and_save_gradual_models(data_path, models_dir='models', static_dir='static/img'): # <-- AÑADIR static_dir
    """
    Entrena modelos, GENERA GRÁFICOS, y guarda los modelos.
    """
    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        return {"error": f"No se pudo leer el archivo Excel: {e}"}

    results = {}
    
    # --- PASO NUEVO: Generar gráficos antes de entrenar ---
    # Usamos una versión limpia del dataframe completo para los gráficos
    df_cleaned_for_plots = clean_data(df.copy())
    # Añadimos la columna 'Estado_nota' original para los gráficos
    df_cleaned_for_plots['Estado_nota'] = df['Estado_nota']
    image_paths = generate_behavioral_plots(df_cleaned_for_plots, static_dir)
    results['image_paths'] = image_paths # Guardamos las rutas para pasarlas a la web
    # ---------------------------------------------------

    # Iterar por cada etapa (UD1 a UD4)
    for stage in range(1, 5):
        print(f"--- Entrenando Modelo para Etapa {stage} ---")
        
        try:
            X, y = preprocess_for_gradual_training(df, stage)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            print(f"Accuracy para Etapa {stage}: {accuracy:.4f}")
            
            model_filename = f'modelo_gradual_ud{stage}.pkl'
            model_path = os.path.join(models_dir, model_filename)
            joblib.dump(model, model_path)
            
            results[f'etapa_{stage}'] = {
                'model_path': model_path,
                'accuracy': accuracy,
                'report': report
            }

        except Exception as e:
            print(f"Error entrenando la etapa {stage}: {e}")
            results[f'etapa_{stage}'] = {'error': str(e)}

    return results