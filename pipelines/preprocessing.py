import pandas as pd
import numpy as np

def load_and_clean_data(filepath, is_training_data=True):
    """
    Carga, limpia y prepara el DataFrame para entrenamiento o predicción.
    Aplica ingeniería de características para mejorar el rendimiento del modelo.

    Parámetros:
    - filepath: Ruta al archivo .xlsx
    - is_training_data: True si contiene PROMEDIO_CURSO, False si es para predicción

    Retorna:
    - df: DataFrame limpio y procesado
    """

    # Definir nombres de columnas esperadas
    column_names = [
        'Periodo', 'CÓDIGO', 'ALUMNO', 'DEPART', 'DISTRITO', 'EDAD', 'MODALIDAD', 'NACIM', 'PROCEDENCIA',
        'PERIODO_INGRESO', 'CARRERA_PLAN', 'DESCRIPCION', 'FECHA_MATRICULA', 'CURSOS_MATRICULADOS',
        'CURSO_CODIGO', 'CURSO_NOMBRE', 'DESCRIPCION_1', 'PROMEDIO_CURSO', 'TIPOSESION', 'SECCIÓN', 'DOCENTE',
        'PERIODO_ACADEMICO', 'PERIODOMES', 'ASISTENCIAS', 'CLASES', 'PORCENTAJE_asistencia', 'Promedio_Final',
        'CUOTA_1', 'CUOTA_2', 'CUOTA_3', 'CUOTA_4', 'CUOTA_5'
    ]

    # Si no es data de entrenamiento, elimina las columnas que no estarán disponibles
    if not is_training_data:
        cols_to_remove = ['PROMEDIO_CURSO', 'Promedio_Final']
        column_names = [col for col in column_names if col not in cols_to_remove]

    # Cargar Excel
    df = pd.read_excel(
        filepath,
        engine='openpyxl',
        header=None,
        skiprows=1,
        names=column_names
    )

    # --- Limpieza y Conversión de Tipos ---
    num_cols = ['PORCENTAJE_asistencia', 'CURSOS_MATRICULADOS', 'EDAD']
    if is_training_data:
        num_cols.append('PROMEDIO_CURSO')

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=num_cols, inplace=True)

    # --- Ingeniería de Características ---

    # 1. Crear columnas derivadas de cuotas pagadas
    cuota_cols = ['CUOTA_1', 'CUOTA_2', 'CUOTA_3', 'CUOTA_4', 'CUOTA_5']
    if all(col in df.columns for col in cuota_cols):
        df['CUOTAS_PAGADAS'] = df[cuota_cols].applymap(lambda x: 1 if str(x).strip().upper() == 'COBRADO' else 0).sum(axis=1)
        df['INDICE_PAGO'] = df['CUOTAS_PAGADAS'] / 5
        df['CUOTAS_PENDIENTES'] = 5 - df['CUOTAS_PAGADAS']


    # 2. Calcular asistencia promedio por curso
    if 'PORCENTAJE_asistencia' in df.columns and 'CURSOS_MATRICULADOS' in df.columns:
        # Evitar división por cero
        df['ASISTENCIA_POR_CURSO'] = df['PORCENTAJE_asistencia'] / df['CURSOS_MATRICULADOS'].replace(0, 1)

    # 3. Crear características de interacción
    if "EDAD" in df.columns and "PORCENTAJE_asistencia" in df.columns:
        df["EDAD_X_ASISTENCIA"] = df["EDAD"] * df["PORCENTAJE_asistencia"]

    if "CURSOS_MATRICULADOS" in df.columns and "INDICE_PAGO" in df.columns:
         df["CURSOS_X_INDICE_PAGO"] = df["CURSOS_MATRICULADOS"] * df["INDICE_PAGO"]

    # 4. Agrupar categorías raras para mejorar el aprendizaje
    if is_training_data:
        cat_cols_to_process = ['PROCEDENCIA', 'DESCRIPCION']
        for col in cat_cols_to_process:
            if col in df.columns:
                value_counts = df[col].value_counts(normalize=True)
                rare_categories = value_counts[value_counts < 0.01].index
                if len(rare_categories) > 0:
                    df[col] = df[col].replace(rare_categories, 'Otros')

    # --- Variable Objetivo ---
    # Crear variable objetivo solo si es data de entrenamiento
    if is_training_data:
        df['ESTADO_APROBACION_NUM'] = np.where(df['PROMEDIO_CURSO'] >= 13, 1, 0)

    return df
