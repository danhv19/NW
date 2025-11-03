import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def preprocess_historical(X_in):
    """
    Preprocesa los datos para el análisis histórico (Proceso 1).
    """
    # Copiar para evitar SettingWithCopyWarning
    X = X_in.copy()

    # Convertir todas las columnas de cuotas a numérico, forzando errores a NaN
    for col in ['CUOTA_1', 'CUOTA_2', 'CUOTA_3', 'CUOTA_4', 'CUOTA_5']:
        if col in X.columns:
            # Asignar 1 a 'COBRADO', 0 a 'PENDIENTE', y NaN a otros
            X[col] = X[col].map({'COBRADO': 1, 'PENDIENTE': 0}).fillna(np.nan)
            
    # Identificar columnas numéricas y categóricas
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    print(f"[Histórico Preprocessing] Numéricas: {numeric_features}")
    print(f"[Histórico Preprocessing] Categóricas: {categorical_features}")

    # --- Creación de Pipelines de Preprocesamiento ---
    
    # Pipeline para variables numéricas:
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)), # Rellenar cuotas/asistencia faltantes con 0
        ('scaler', StandardScaler())
    ])

    # Pipeline para variables categóricas:
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # --- Combinar Pipelines con ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' 
    )

    X_processed = preprocessor.fit_transform(X)
    
    print("Preprocesamiento histórico completado.")
    return X_processed, preprocessor
