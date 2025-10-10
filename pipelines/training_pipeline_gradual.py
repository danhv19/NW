import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import shutil

from pipelines.preprocessing_gradual import preprocess_for_gradual_training, clean_data
from pipelines.plotting_gradual import generate_behavioral_plots

def train_and_save_gradual_models(data_path, models_dir='models', static_dir='static/img'):
    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        return {"error": f"No se pudo leer el archivo Excel: {e}"}

    results = {}
    
    df_cleaned_for_plots = clean_data(df.copy())
    df_cleaned_for_plots['Estado_nota'] = df['Estado_nota']
    image_paths = generate_behavioral_plots(df_cleaned_for_plots, static_dir)
    results['image_paths'] = image_paths

    # --- NUEVO: Variables para rastrear el mejor modelo ---
    best_overall_accuracy = 0.0
    best_model_path = ""
    best_model_stage = 0
    # ----------------------------------------------------

    for stage in range(1, 5):
        try:
            X, y = preprocess_for_gradual_training(df, stage)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            model_filename = f'modelo_gradual_ud{stage}.pkl'
            model_path = os.path.join(models_dir, model_filename)
            joblib.dump(model, model_path)

            # --- NUEVO: Comparamos para encontrar el mejor modelo ---
            if accuracy > best_overall_accuracy:
                best_overall_accuracy = accuracy
                best_model_path = model_path
                best_model_stage = stage
            # -------------------------------------------------------
            
            results[f'etapa_{stage}'] = {
                'model_path': model_path,
                'accuracy': accuracy,
                'report': report
            }
        except Exception as e:
            results[f'etapa_{stage}'] = {'error': str(e)}

    # --- NUEVO: Guardamos una copia del mejor modelo con un nombre genérico y lo añadimos a los resultados ---
    if best_model_path:
        best_model_generic_name = 'mejor_modelo_gradual.pkl'
        best_model_generic_path = os.path.join(models_dir, best_model_generic_name)
        shutil.copy(best_model_path, best_model_generic_path)
        results['best_model'] = {
            'path': best_model_generic_path,
            'stage': best_model_stage,
            'accuracy': best_overall_accuracy
        }
    # ------------------------------------------------------------------------------------------------------

    return results