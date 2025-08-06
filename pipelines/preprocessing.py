import pandas as pd
import numpy as np

def load_and_clean_data(filepath, is_training_data=True):
    """
    Carga, limpia y prepara el DataFrame para entrenamiento o predicción.

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

    # Cargar Excel (sin encabezado real, por eso se usa skiprows=1)
    df = pd.read_excel(
        filepath,
        engine='openpyxl',
        header=None,
        skiprows=1,
        names=column_names
    )

    # Conversión de columnas numéricas
    num_cols = ['PORCENTAJE_asistencia', 'CURSOS_MATRICULADOS', 'EDAD']
    if is_training_data:
        num_cols.append('PROMEDIO_CURSO')

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=num_cols, inplace=True)

    # Crear columnas derivadas de cuotas pagadas
    cuota_cols = ['CUOTA_1', 'CUOTA_2', 'CUOTA_3', 'CUOTA_4', 'CUOTA_5']
    if all(col in df.columns for col in cuota_cols):
        df['CUOTAS_PAGADAS'] = df[cuota_cols].applymap(lambda x: 1 if str(x).strip().upper() == 'COBRADO' else 0).sum(axis=1)
        df['INDICE_PAGO'] = df['CUOTAS_PAGADAS'] / 5

    # Calcular asistencia promedio por curso
    if 'PORCENTAJE_asistencia' in df.columns and 'CURSOS_MATRICULADOS' in df.columns:
        df['ASISTENCIA_POR_CURSO'] = df['PORCENTAJE_asistencia'] / df['CURSOS_MATRICULADOS']

    # Crear variable objetivo si es entrenamiento
    if is_training_data:
        df['ESTADO_APROBACION_NUM'] = np.where(df['PROMEDIO_CURSO'] >= 13, 1, 0)

    return df
