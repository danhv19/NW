import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def preprocess_historical(X):
    """
    Preprocesa los datos para el análisis histórico.
    """
    
    # Convertir todas las columnas de cuotas a numérico, forzando errores a NaN
    for col in ['CUOTA_1', 'CUOTA_2', 'CUOTA_3', 'CUOTA_4', 'CUOTA_5']:
        if col in X.columns:
            # Asignar 1 a 'COBRADO', 0 a 'PENDIENTE', y NaN a otros
            X[col] = X[col].map({'COBRADO': 1, 'PENDIENTE': 0}).fillna(np.nan)
            
    # Identificar columnas numéricas y categóricas
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    # --- Creación de Pipelines de Preprocesamiento ---
    
    # Pipeline para variables numéricas:
    # 1. Imputar (rellenar valores faltantes, ej. en cuotas) con 0 (significa PENDIENTE)
    # 2. Escalar los datos (importante para Regresión Logística)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    # Pipeline para variables categóricas:
    # 1. Imputar (rellenar valores faltantes) con una categoría 'Desconocido'
    # 2. Aplicar One-Hot Encoding (convertir 'Lima', 'Callao' en columnas 0/1)
    #    handle_unknown='ignore' es clave para que la predicción no falle si ve un distrito nuevo.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # --- Combinar Pipelines con ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Dejar pasar columnas no especificadas (aunque no debería haber)
    )

    # Aplicar el preprocesador
    X_processed = preprocessor.fit_transform(X)
    
    print("Preprocesamiento histórico completado.")
    # Devolver los datos procesados Y el preprocesador (para guardarlo)
    return X_processed, preprocessor
