import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def standardize_gradual_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Homologa los encabezados usados durante el entrenamiento y la predicción."""

    special_case_map = {
        'Periodo': 'Periodo',
        'PERIODO': 'Periodo_Academico'
    }

    generic_map = {
        'CARRERA': 'Carrera',
        'CURSO': 'Curso',
        'CÓDIGO': 'Codigo',
        'CODIGO': 'Codigo',
        'EDAD': 'Edad',
        'DOCENTE': 'Docente',
        'MODALIDAD': 'Modalidad',
        'TIPOSESION': 'Tiposesion',
        'SECCIÓN': 'Seccion',
        'SECCION': 'Seccion',
        'ASISTENCIAS': 'Asistencias',
        'PORCENTAJE_ASISTENCIA': 'Asistencia',
        'ASISTENCIA': 'Asistencia',
        'CUOTA_1': 'CUOTA_1',
        'CUOTA_2': 'CUOTA_2',
        'CUOTA_3': 'CUOTA_3',
        'CUOTA_4': 'CUOTA_4',
        'CUOTA_5': 'CUOTA_5',
        'U1': 'ud1',
        'U2': 'ud2',
        'U3': 'ud3',
        'U4': 'ud4',
        'ESTADO_NOTA': 'Estado_nota',
        'ESTADO': 'Estado_nota',
        'PROMEDIO_CURSO': 'promedio_curso',
        'FECHA_MATRICULA': 'fecha_matricula',
        'FECHA_INICIO': 'fecha_inicio',
        'FECHA_FIN': 'fecha_fin',
        'PROMEDIO': 'promedio'
    }

    new_columns = []
    seen = {}
    for col in df.columns:
        original = col.strip()
        if original in special_case_map:
            new_col = special_case_map[original]
        else:
            key = original.upper()
            new_col = generic_map.get(key, original)

        if new_col in seen:
            seen[new_col] += 1
            new_col = f"{new_col}_{seen[new_col]}"
        else:
            seen[new_col] = 0

        new_columns.append(new_col)

    df.columns = new_columns
    return df

def clean_data(df):
    """
    Realiza la limpieza básica de datos.
    """
    if 'Asistencia' in df.columns:
        if df['Asistencia'].dtype == 'object':
            df['Asistencia'] = df['Asistencia'].str.replace('%', '', regex=False)
        df['Asistencia'] = pd.to_numeric(df['Asistencia'], errors='coerce')
        if df['Asistencia'].max(skipna=True) is not None and df['Asistencia'].max(skipna=True) > 1:
            df['Asistencia'] = df['Asistencia'] / 100.0

    if 'Asistencias' in df.columns:
        df['Asistencias'] = pd.to_numeric(df['Asistencias'], errors='coerce').fillna(0)

    for col in ['fecha_matricula', 'fecha_inicio', 'fecha_fin']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    for col in ['CUOTA_1', 'CUOTA_2', 'CUOTA_3', 'CUOTA_4', 'CUOTA_5']:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .map({'COBRADO': 1, 'PENDIENTE': 0})
                .fillna(0)
            )

    for col in ['ud1', 'ud2', 'ud3', 'ud4']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Estado_nota' in df.columns:
        df['target'] = df['Estado_nota'].map({'Aprobado': 1, 'Desaprobado': 0})
        
    return df

def feature_engineering_by_stage(df, stage):
    """
    Crea características específicas para cada etapa de forma segura.
    """
    if 'fecha_matricula' in df.columns and 'fecha_inicio' in df.columns:
        df['dias_anticipacion_matricula'] = (df['fecha_inicio'] - df['fecha_matricula']).dt.days
    else:
        df['dias_anticipacion_matricula'] = 0

    if stage >= 1:
        df['cuotas_pagadas_ud1'] = df.get('CUOTA_1', 0)
    if stage >= 2:
        df['cuotas_pagadas_ud2'] = df.get('CUOTA_1', 0) + df.get('CUOTA_2', 0)
    if stage >= 3:
        df['cuotas_pagadas_ud3'] = df.get('CUOTA_1', 0) + df.get('CUOTA_2', 0) + df.get('CUOTA_3', 0)
        df['mejora_ud2'] = df.get('ud2', 0) - df.get('ud1', 0)
    if stage >= 4:
        df['cuotas_pagadas_ud4'] = df.get('CUOTA_1', 0) + df.get('CUOTA_2', 0) + df.get('CUOTA_3', 0) + df.get('CUOTA_4', 0)
        df['mejora_ud3'] = df.get('ud3', 0) - df.get('ud2', 0)
        
    return df

def get_features_for_stage(stage):
    """
    Define las características correctas para cada ETAPA DE PREDICCIÓN.
    """
    base_features = [
        'Periodo',
        'Periodo_Academico',
        'Carrera',
        'Curso',
        'Docente',
        'Modalidad',
        'Tiposesion',
        'Seccion',
        'Edad',
        'Asistencia',
        'Asistencias',
        'dias_anticipacion_matricula'
    ]
    
    # Modelo 1: Se entrena SIN NOTAS. Se usa para predecir ANTES de la UD1.
    if stage == 1:
        return base_features + ['cuotas_pagadas_ud1']
        
    # Modelo 2: Se entrena CON la nota de UD1. Se usa para predecir DESPUÉS de la UD1.
    elif stage == 2:
        return base_features + ['cuotas_pagadas_ud2', 'ud1']
        
    # Modelo 3: Se entrena CON las notas de UD1 y UD2. Se usa para predecir DESPUÉS de la UD2.
    elif stage == 3:
        return base_features + ['cuotas_pagadas_ud3', 'ud1', 'ud2', 'mejora_ud2']
        
    # Modelo 4: Se entrena CON las notas de UD1, UD2 y UD3. Se usa para predecir DESPUÉS de la UD3.
    elif stage == 4:
        return base_features + ['cuotas_pagadas_ud4', 'ud1', 'ud2', 'ud3', 'mejora_ud2', 'mejora_ud3']
        
    return []


_CATEGORICAL_EXPECTED = {
    'Periodo',
    'Periodo_Academico',
    'Carrera',
    'Curso',
    'Docente',
    'Modalidad',
    'Tiposesion',
    'Seccion',
}


def ensure_expected_features(df: pd.DataFrame, stage: int) -> pd.DataFrame:
    """Guarantee that every expected feature for the stage exists with a safe default."""

    expected = get_features_for_stage(stage)

    for col in expected:
        if col not in df.columns:
            if col in _CATEGORICAL_EXPECTED:
                df[col] = 'Desconocido'
            else:
                df[col] = 0

    return df[expected]

def preprocess_for_gradual_training(df, stage):
    """
    Orquesta el preprocesamiento para el entrenamiento y devuelve los datos y el preprocesador.
    """
    df_standardized = standardize_gradual_columns(df.copy())
    df_clean = clean_data(df_standardized)
    df_featured = feature_engineering_by_stage(df_clean, stage)

    if 'target' not in df_featured.columns:
        raise ValueError("La columna 'Estado_nota' (mapeada a 'target') no se encontró.")

    X = ensure_expected_features(df_featured, stage)
    y = df_featured['target']

    numeric_features = X.select_dtypes(include='number').columns.tolist()
    categorical_features = X.select_dtypes(exclude='number').columns.tolist()

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, preprocessor