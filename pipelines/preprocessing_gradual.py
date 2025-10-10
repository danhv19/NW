import pandas as pd

def clean_data(df):
    """
    Realiza la limpieza básica de datos.
    """
    if 'Asistencia' in df.columns and df['Asistencia'].dtype == 'object':
        df['Asistencia'] = df['Asistencia'].str.replace('%', '', regex=False).astype(float) / 100.0

    for col in ['fecha_matricula', 'fecha_inicio', 'fecha_fin']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
            
    for col in ['CUOTA_1', 'CUOTA_2', 'CUOTA_3', 'CUOTA_4']:
        if col in df.columns:
            df[col] = df[col].map({'COBRADO': 1, 'PENDIENTE': 0}).fillna(0)

    if 'Estado_nota' in df.columns:
        df['target'] = df['Estado_nota'].map({'Aprobado': 1, 'Desaprobado': 0})
        # No eliminamos 'Estado_nota' para poder usarla en los gráficos
        
    return df

def feature_engineering_by_stage(df, stage):
    """
    Crea características específicas para cada etapa (1, 2, 3 o 4).
    """
    if 'fecha_matricula' in df.columns and 'fecha_inicio' in df.columns:
        df['dias_anticipacion_matricula'] = (df['fecha_inicio'] - df['fecha_matricula']).dt.days

    if stage >= 1:
        df['cuotas_pagadas_ud1'] = df['CUOTA_1']
    if stage >= 2:
        df['cuotas_pagadas_ud2'] = df['CUOTA_1'] + df['CUOTA_2']
        df['mejora_ud2'] = df['ud2'] - df['ud1']
    if stage >= 3:
        df['cuotas_pagadas_ud3'] = df['CUOTA_1'] + df['CUOTA_2'] + df['CUOTA_3']
        df['mejora_ud3'] = df['ud3'] - df['ud2']
    if stage >= 4:
        df['cuotas_pagadas_ud4'] = df['CUOTA_1'] + df['CUOTA_2'] + df['CUOTA_3'] + df['CUOTA_4']
        df['mejora_ud4'] = df['ud4'] - df['promedio'] 
        
    return df

def get_features_for_stage(stage):
    """
    Devuelve la lista de columnas a usar como características para cada etapa.
    """
    base_features = ['Carrera', 'Docente', 'Ciclo', 'dias_anticipacion_matricula']
    
    if stage == 1:
        return base_features + ['ud1', 'cuotas_pagadas_ud1', 'Asistencia']
    elif stage == 2:
        return base_features + ['ud1', 'ud2', 'cuotas_pagadas_ud2', 'Asistencia', 'mejora_ud2']
    elif stage == 3:
        return base_features + ['ud1', 'ud2', 'ud3', 'cuotas_pagadas_ud3', 'Asistencia', 'mejora_ud3']
    elif stage == 4:
        return base_features + ['ud1', 'ud2', 'ud3', 'ud4', 'promedio', 'cuotas_pagadas_ud4', 'Asistencia', 'mejora_ud4']
    else:
        return []

def preprocess_for_gradual_training(df, stage):
    """
    Función principal que orquesta el preprocesamiento para una etapa específica.
    """
    df_clean = clean_data(df.copy())
    df_featured = feature_engineering_by_stage(df_clean, stage)
    
    features = get_features_for_stage(stage)
    
    if 'target' not in df_featured.columns:
        raise ValueError("La columna 'Estado_nota' (mapeada a 'target') no se encontró.")
        
    X = df_featured[features]
    y = df_featured['target']
    
    X = pd.get_dummies(X, columns=['Carrera', 'Docente'], drop_first=True)
    
    return X, y