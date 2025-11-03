import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def preprocess_gradual(X):
    """
    Preprocesa los datos para los modelos graduales.
    Detecta automáticamente las columnas numéricas y categóricas.
    """
    
    # Identificar columnas numéricas y categóricas
    # Características de notas que podrían estar presentes
    note_features = ['U1', 'U2', 'U3', 'U4']
    
    # Características de pago
    payment_features = ['CUOTA_1', 'CUOTA_2', 'CUOTA_3', 'CUOTA_4', 'CUOTA_5']
    
    # Otras características numéricas conocidas
    other_numeric = ['EDAD', 'PERIODO', 'ASISTENCIAS', 'PORCENTAJE_asistencia']
    
    # Construir lista de todas las posibles características numéricas
    numeric_features = [col for col in X.columns if col in note_features + payment_features + other_numeric]
    
    # Las características categóricas son todas las que no son numéricas
    categorical_features = [col for col in X.columns if col not in numeric_features]

    print(f"[Preprocesador] Numéricas detectadas: {numeric_features}")
    print(f"[Preprocesador] Categóricas detectadas: {categorical_features}")

    # --- Crear Pipelines de Transformación ---

    # Pipeline para datos numéricos:
    # 1. Imputar valores faltantes (NaN) con la mediana.
    # 2. Escalar los datos para que tengan media 0 y desviación 1.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para datos categóricos:
    # 1. Imputar valores faltantes con la etiqueta 'Desconocido'.
    # 2. Aplicar One-Hot Encoding (convertir 'Carrera_A', 'Carrera_B' en columnas 0/1).
    #    handle_unknown='ignore' es crucial para que la predicción no falle si ve una categoría nueva.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # --- Combinar Pipelines con ColumnTransformer ---
    # Este objeto aplica el pipeline correcto a la columna correcta.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Dejar pasar cualquier columna no especificada (aunque no debería haber)
    )

    # Ajustar el preprocesador y transformar los datos
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, preprocessor
