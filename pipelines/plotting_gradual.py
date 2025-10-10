import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_behavioral_plots(df, static_folder='static/img'):
    """
    Genera y guarda gráficos que muestran el comportamiento evolutivo de los estudiantes.
    """
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)
        
    image_paths = {}
    
    df_plot = df.copy()
    if 'Estado_nota' not in df_plot.columns:
        if 'target' in df_plot.columns:
            df_plot['Estado_nota'] = df_plot['target'].map({1: 'Aprobado', 0: 'Desaprobado'})
        else:
            return {}

    # --- GRÁFICO 1: Evolución del Rendimiento Promedio ---
    try:
        plt.figure(figsize=(10, 6))
        df_melted = df_plot.melt(id_vars=['Estado_nota'], value_vars=['ud1', 'ud2', 'ud3', 'promedio'], 
                                 var_name='Etapa', value_name='Nota')
        sns.lineplot(data=df_melted, x='Etapa', y='Nota', hue='Estado_nota', marker='o', errorbar=None)
        plt.title('Evolución de Notas Promedio (Aprobados vs. Desaprobados)')
        plt.ylabel('Nota Promedio')
        plt.grid(True)
        img_path = os.path.join(static_folder, 'gradual_plot_rendimiento.png')
        plt.savefig(img_path)
        plt.close()
        image_paths['Evolucion_Rendimiento'] = img_path
    except Exception as e:
        print(f"Error al generar gráfico de rendimiento: {e}")

    # --- GRÁFICO 2: Tasa de Aprobación vs. Cuotas Pagadas a la UD3 ---
    try:
        plt.figure(figsize=(10, 6))
        # Asegurarse que las columnas de cuotas sean numéricas
        for col in ['CUOTA_1', 'CUOTA_2', 'CUOTA_3']:
            if col not in df_plot.columns: df_plot[col] = 0 # Añadir si no existe
        df_plot['cuotas_pagadas_ud3'] = df_plot['CUOTA_1'] + df_plot['CUOTA_2'] + df_plot['CUOTA_3']
        
        # 'target' debe ser numérico (0 o 1) para calcular el promedio (tasa)
        if 'target' not in df_plot.columns: df_plot['target'] = df_plot['Estado_nota'].map({'Aprobado': 1, 'Desaprobado': 0})

        sns.barplot(data=df_plot, x='cuotas_pagadas_ud3', y='target', palette='viridis')
        plt.title('Tasa de Aprobación según Cuotas Pagadas (hasta UD3)')
        plt.xlabel('Número de Cuotas Pagadas al finalizar la UD3')
        plt.ylabel('Tasa de Aprobación Promedio')
        plt.grid(axis='y')
        img_path = os.path.join(static_folder, 'gradual_plot_pagos.png')
        plt.savefig(img_path)
        plt.close()
        image_paths['Impacto_Pagos'] = img_path
    except Exception as e:
        print(f"Error al generar gráfico de pagos: {e}")
        
    return image_paths