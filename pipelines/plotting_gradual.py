import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Desactivar advertencias de futuras versiones
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def plot_gradual_results(df, target_column, target_unit, plots_folder):
    """
    Genera y guarda una serie de gráficos de Análisis Exploratorio de Datos (EDA)
    para la fase de entrenamiento gradual.
    """
    print(f"Generando gráficos para el objetivo: {target_column} (Unidad: {target_unit})")
    
    # Asegurarse de que la carpeta de gráficos exista
    os.makedirs(plots_folder, exist_ok=True)
    
    # Lista para almacenar las rutas de los gráficos generados
    plot_paths = {}

    # --- Gráfico 1: Distribución del Promedio Objetivo ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_column], kde=True, bins=20)
    plt.axvline(10.5, color='red', linestyle='--', label='Límite Aprobado (10.5)')
    plt.title(f'Distribución de Notas para {target_column}')
    plt.xlabel('Nota')
    plt.ylabel('Frecuencia')
    plt.legend()
    plot_name = f'eda_distribucion_{target_unit}.png'
    plot_path = os.path.join(plots_folder, plot_name)
    plt.savefig(plot_path)
    plt.close()
    plot_paths['distribucion'] = plot_name

    # --- Gráfico 2: Correlación de Variables Numéricas ---
    # Seleccionar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(15, 10))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
        plt.title('Mapa de Calor de Correlaciones Numéricas')
        plot_name = f'eda_heatmap_corr_{target_unit}.png'
        plot_path = os.path.join(plots_folder, plot_name)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        plot_paths['heatmap'] = plot_name

    # --- Gráfico 3: Impacto de 'Carrera' en el Promedio ---
    if 'Carrera' in df.columns:
        plt.figure(figsize=(12, 8))
        # Tomar las 10 carreras más frecuentes para evitar gráficos saturados
        top_carreras = df['Carrera'].value_counts().nlargest(10).index
        df_top_carreras = df[df['Carrera'].isin(top_carreras)]
        
        sns.boxplot(data=df_top_carreras, x=target_column, y='Carrera', palette='viridis')
        plt.axvline(10.5, color='red', linestyle='--', label='Límite Aprobado (10.5)')
        plt.title(f'Distribución de {target_column} por Carrera (Top 10)')
        plt.xlabel(f'Nota {target_column}')
        plt.ylabel('Carrera')
        plt.legend()
        plot_name = f'eda_box_carrera_{target_unit}.png'
        plot_path = os.path.join(plots_folder, plot_name)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        plot_paths['box_carrera'] = plot_name

    # --- Gráfico 4: Impacto de 'PORCENTAJE_asistencia' ---
    if 'PORCENTAJE_asistencia' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='PORCENTAJE_asistencia', y=target_column, hue='TARGET_Aprobado', palette='seismic', alpha=0.6)
        plt.axvline(df['PORCENTAJE_asistencia'].median(), color='orange', linestyle='--', label='Mediana Asistencia')
        plt.axhline(10.5, color='red', linestyle='--', label='Límite Aprobado (10.5)')
        plt.title(f'Asistencia vs. Nota {target_column}')
        plt.xlabel('Porcentaje de Asistencia')
        plt.ylabel(f'Nota {target_column}')
        plt.legend()
        plot_name = f'eda_scatter_asistencia_{target_unit}.png'
        plot_path = os.path.join(plots_folder, plot_name)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        plot_paths['scatter_asistencia'] = plot_name

    print(f"Gráficos generados: {list(plot_paths.keys())}")
    return plot_paths
