import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from pipelines.training_pipeline_gradual import run_gradual_training
from pipelines.prediction_pipeline_gradual import run_gradual_prediction
import pandas as pd

# --- Configuración de Carpetas ---
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER_TRAIN = os.path.join(BASE_DIR, 'uploads', 'training')
UPLOAD_FOLDER_PREDICT = os.path.join(BASE_DIR, 'uploads', 'prediction')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
STATIC_IMG_FOLDER = os.path.join(BASE_DIR, 'static', 'img')

app.config['UPLOAD_FOLDER_TRAIN'] = UPLOAD_FOLDER_TRAIN
app.config['UPLOAD_FOLDER_PREDICT'] = UPLOAD_FOLDER_PREDICT
app.config['MODELS_FOLDER'] = MODELS_FOLDER
app.config['STATIC_IMG_FOLDER'] = STATIC_IMG_FOLDER
app.config['SECRET_KEY'] = 'my_secret_key_123' # Necesario para 'flash'

# Asegurarse de que las carpetas existan
os.makedirs(UPLOAD_FOLDER_TRAIN, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_PREDICT, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(STATIC_IMG_FOLDER, exist_ok=True)

# --- Rutas Principales ---

@app.route('/')
def index():
    """Página de inicio que muestra las opciones."""
    return render_template('index.html')

# --- Ruta de Entrenamiento Gradual ---

@app.route('/train_gradual', methods=['GET', 'POST'])
def train_gradual_model():
    """
    Entrena un modelo específico para una unidad (U1, U2, U3, U4)
    basado en los datos históricos (ej: Data - consolidada.xlsx).
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No se seleccionó ningún archivo', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        target_unit = request.form.get('target_unit') # Obtiene la unidad (U1, U2, etc.) del formulario
        
        if file.filename == '':
            flash('No se seleccionó ningún archivo', 'danger')
            return redirect(request.url)
        
        if not target_unit:
            flash('No se seleccionó la unidad a entrenar (U1, U2, U3, U4)', 'danger')
            return redirect(request.url)
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER_TRAIN'], filename)
            file.save(filepath)
            
            try:
                # --- ACTUALIZACIÓN ---
                # Se eliminó 'plots' de los valores de retorno para solucionar el ImportError
                metrics, model_filename = run_gradual_training(
                    filepath, 
                    target_unit, 
                    app.config['MODELS_FOLDER'], 
                    app.config['STATIC_IMG_FOLDER']
                )
                flash(f'Modelo para {target_unit} entrenado y guardado como {model_filename}!', 'success')
                return render_template('train_results_gradual.html', 
                                       metrics=metrics, 
                                       # plots=plots, # Desactivado temporalmente
                                       model_filename=model_filename, 
                                       target_unit=target_unit)
            except Exception as e:
                flash(f'Error durante el entrenamiento gradual: {str(e)}', 'danger')
                return redirect(request.url)
    
    # Si es GET, muestra la página de subida para entrenamiento gradual
    return render_template('train_gradual.html')

# --- Ruta de Predicción Gradual ---

@app.route('/predict_gradual', methods=['GET', 'POST'])
def predict_gradual():
    """
    Detecta automáticamente qué unidad predecir (U1, U2, U3, U4)
    basado en las columnas de notas presentes en el archivo subido.
    (ej: Para predecir U1.xlsx, Para predecir U2.xlsx)
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No se seleccionó ningún archivo', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No se seleccionó ningún archivo', 'danger')
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER_PREDICT'], filename)
            file.save(filepath)
            
            try:
                # Ejecutar el pipeline de predicción gradual "inteligente"
                results_df, results_filename, prediction_target = run_gradual_prediction(
                    filepath, 
                    app.config['MODELS_FOLDER'], 
                    app.config['UPLOAD_FOLDER_PREDICT']
                )
                
                flash(f'Predicción para {prediction_target} completada.', 'success')
                return render_template('prediction_results_gradual.html', 
                                       tables=[results_df.to_html(classes='table table-striped table-hover', header="true", index=False)], 
                                       results_path=results_filename,
                                       prediction_target=prediction_target)
            except Exception as e:
                flash(f'Error durante la predicción gradual: {str(e)}', 'danger')
                return redirect(request.url)
    
    # Si es GET, solo muestra la página de subida
    return render_template('predict_gradual.html')

# --- Ruta para descargar resultados ---

@app.route('/uploads/prediction/<filename>')
def uploaded_file(filename):
    """Permite descargar el archivo CSV de resultados de predicción."""
    return send_from_directory(app.config['UPLOAD_FOLDER_PREDICT'], 
                               filename, 
                               as_attachment=True)

# --- Rutas para mostrar imágenes de EDA ---
# (Desactivada temporalmente mientras se corrige plotting_gradual.py)
# @app.route('/static/img/<filename>')
# def send_img(filename):
#    """Sirve las imágenes de gráficos generadas."""
#    return send_from_directory(app.config['STATIC_IMG_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
