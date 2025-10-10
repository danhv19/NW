import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os

from pipelines.preprocessing_gradual import preprocess_for_gradual_training
from pipelines.plotting_gradual import generate_behavioral_plots

def train_and_save_gradual_models(data_path, models_dir='models', static_dir='static/img'):
    """
    Entrena, compara y guarda el mejor modelo para cada una de las 4 etapas.
    """
    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        return {"error": f"No se pudo leer el archivo Excel: {e}"}

    results = {}
    
    try:
        image_paths = generate_behavioral_plots(df.copy(), static_dir)
        results['image_paths'] = image_paths
    except Exception as e:
        print(f"Error generando gráficos: {e}")
        results['image_paths'] = {}

    for stage in range(1, 5):
        print(f"\n--- PROCESANDO ETAPA {stage} ---")
        try:
            X_train, X_test, y_train, y_test, preprocessor = preprocess_for_gradual_training(df, stage)

            models_to_test = {
                "Regresión Logística": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
                "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100)
            }
            
            best_accuracy_for_stage = 0.0
            best_pipeline_for_stage = None
            best_model_name_for_stage = ""

            for model_name, model in models_to_test.items():
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])

                pipeline.fit(X_train, y_train)
                
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"  - Modelo: {model_name}, Accuracy: {accuracy:.4f}")

                if accuracy > best_accuracy_for_stage:
                    best_accuracy_for_stage = accuracy
                    best_pipeline_for_stage = pipeline
                    best_model_name_for_stage = model_name
            
            if best_pipeline_for_stage:
                print(f"🏆 Ganador de la Etapa {stage}: {best_model_name_for_stage} con Accuracy de {best_accuracy_for_stage:.4f}")
                model_filename = f'modelo_gradual_ud{stage}.pkl'
                model_path = os.path.join(models_dir, model_filename)
                joblib.dump(best_pipeline_for_stage, model_path)
                
                results[f'Etapa {stage}'] = {
                    'model_path': model_path,
                    'accuracy': best_accuracy_for_stage,
                    'best_model_name': best_model_name_for_stage
                }
            else:
                results[f'Etapa {stage}'] = {'error': 'No se pudo determinar un mejor modelo.'}

        except Exception as e:
            print(f"Error procesando la Etapa {stage}: {e}")
            results[f'Etapa {stage}'] = {'error': str(e)}

    return results