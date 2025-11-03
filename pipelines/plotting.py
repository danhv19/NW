import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Configuración de estilo de Seaborn
sns.set_theme(style="whitegrid")

def plot_historical_results(df, target_col, features, plots_folder):
    """
    Genera y guarda una serie de gráficos de Análisis Exploratorio de Datos (EDA)
    para el pipeline de análisis histórico.
    """
    print("Iniciando generación de gráficos...")
    plots = {}
    
    # Asegurarse de que el target sea numérico para los gráficos
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

    # 1. Gráfico de Correlación (Heatmap)
    try:
        plt.figure(figsize=(16, 10))
        numeric_cols = df[features].select_dtypes(include=np.number).columns
        corr = df[numeric_cols].corrwith(df[target_col]).sort_values(ascending=False).to_frame()
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt='.2f')
        plt.title('Correlación de Variables Numéricas con el Resultado (Aprobado/Desaprobado)')
        plot_filename = 'eda_heatmap_corr.png'
        plt.savefig(os.path.join(plots_folder, plot_filename))
        plt.close()
        plots['correlacion_heatmap'] = plot_filename
    except Exception as e:
        print(f"Error generando heatmap: {e}")

    # 2. Gráfico de Asistencia vs. Resultado
    if 'PORCENTAJE_asistencia' in df.columns:
        try:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=target_col, y='PORCENTAJE_asistencia', data=df)
            plt.title('Distribución de Asistencia vs. Resultado')
            plt.xlabel('Resultado (0 = Desaprobado, 1 = Aprobado)')
            plt.ylabel('Porcentaje de Asistencia')
            plot_filename = 'eda_box_asistencia.png'
            plt.savefig(os.path.join(plots_folder, plot_filename))
            plt.close()
            plots['asistencia_vs_resultado'] = plot_filename
        except Exception as e:
            print(f"Error generando gráfico de asistencia: {e}")

    # 3. Gráfico de Carrera vs. Resultado
    if 'Carrera' in df.columns:
        try:
            plt.figure(figsize=(12, 8))
            # Calcular el promedio de aprobación por carrera
            career_success = df.groupby('Carrera')[target_col].mean().sort_values(ascending=False)
            sns.barplot(x=career_success.values, y=career_success.index, palette="viridis")
            plt.title('Tasa de Aprobación por Carrera')
            plt.xlabel('Tasa de Aprobación Promedio')
            plt.ylabel('Carrera')
            plt.tight_layout()
            plot_filename = 'eda_bar_carrera.png'
            plt.savefig(os.path.join(plots_folder, plot_filename))
            plt.close()
            plots['tasa_aprobacion_carrera'] = plot_filename
        except Exception as e:
            print(f"Error generando gráfico de carrera: {e}")

    # 4. Gráfico de Edad vs. Resultado
    if 'EDAD' in df.columns:
        try:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(df[df[target_col] == 1]['EDAD'], label='Aprobado', fill=True)
            sns.kdeplot(df[df[target_col] == 0]['EDAD'], label='Desaprobado', fill=True)
            plt.title('Distribución de Edad por Resultado')
            plt.xlabel('Edad')
            plt.legend()
            plot_filename = 'eda_kde_edad.png'
            plt.savefig(os.path.join(plots_folder, plot_filename))
            plt.close()
            plots['distribucion_edad'] = plot_filename
        except Exception as e:
            print(f"Error generando gráfico de edad: {e}")
            
    print(f"Gráficos generados: {list(plots.keys())}")
    return plots
