import pandas as pd
import joblib
from pipelines.preprocessing_gradual import preprocess_for_gradual_prediction

def run_gradual_prediction(data_path, model_path):
    """
    Realiza predicciones en datos nuevos usando un modelo gradual entrenado.
    """
    try:
        model = joblib.load(model_path)
        df_new = pd.read_excel(data_path)
        df_original = df_new.copy()

        # Preprocesar los datos para la predicción
        X_processed = preprocess_for_gradual_prediction(df_new)
        
        # Alinear columnas con las del modelo (muy importante)
        model_cols = model.named_steps['preprocessor'].get_feature_names_out()
        
        # Para las dummies que se crearon en el entrenamiento pero no en la predicción
        for col in model_cols:
            if col not in X_processed.columns:
                X_processed[col] = 0
        
        # Reordenar para que coincidan exactamente
        X_processed = X_processed[model_cols]

        # Realizar predicciones
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)[:, 1]

        df_original['PREDICCION'] = ['Aprobado' if p == 1 else 'Desaprobado' for p in predictions]
        df_original['PROBABILIDAD_APROBAR'] = [f"{p:.2%}" for p in probabilities]
        
        return df_original, None # Devolvemos None para el path del excel, se puede implementar después

    except Exception as e:
        print(f"Error durante la predicción gradual: {e}")
        return None, str(e)