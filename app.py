import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import pandas as pd

# Importar los pipelines
from pipelines.historical_analysis_pipeline import run_historical_analysis
from pipelines.training_pipeline_gradual import run_gradual_training
from pipelines.prediction_pipeline_gradual import run_gradual_prediction

# --- Configuración de Carpetas ---
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
STATIC_IMG_FOLDER = os.path.join(BASE_DIR, 'static', 'img')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODELS_FOLDER'] = MODELS_FOLDER
app.config['STATIC_IMG_FOLDER'] = STATIC_IMG_FOLDER
app.config['SECRET_KEY'] = 'my_secret_key_123' 

# Asegurarse de que las carpetas existan
os.makedirs(os.path.join(UPLOAD_FOLDER, 'training'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'prediction'), exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(STATIC_IMG_FOLDER, exist_ok=True)

# --- Rutas Principales ---

@app.route('/')
def index():
    """Página de inicio (Dashboard)"""
    return render_template('index.html')

# --- PROCESO 1: ANÁLISIS HISTÓRICO ---

@app.route('/analysis', methods=['GET', 'POST'])
def historical_analysis():
    """
    Página para el Proceso 1: Análisis Histórico Profundo.
    Usa el "Data - consolidada.xlsx" para analizar el PROMEDIO_CURSO final.
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
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'training', filename)
            file.save(filepath)
            
            try:
                metrics, plots, model_filename = run_historical_analysis(
                    filepath, 
                    app.config['MODELS_FOLDER'], 
                    app.config['STATIC_IMG_FOLDER']
                )
                flash(f'Análisis histórico completado. Mejor modelo: {model_filename}!', 'success')
                return render_template('analysis_results.html', 
                                       metrics=metrics, 
                                       plots=plots, 
                                       model_filename=model_filename)
            except Exception as e:
                flash(f'Error durante el análisis histórico: {str(e)}', 'danger')
                return redirect(request.url)
    
    # Si es GET, muestra la página de subida
    return render_template('analysis.html')

# --- PROCESO 2: SISTEMA GRADUAL (SAT) ---

@app.route('/train_gradual', methods=['GET', 'POST'])
def train_gradual_model():
    """
    Página para entrenar los modelos de unidad (U1, U2, U3, U4).
    Usa el "Data - consolidada.xlsx" (que debe tener U1-U4) para crear
    los modelos pkl graduales.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No se seleccionó ningún archivo', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        target_unit = request.form.get('target_unit') 
        
        if file.filename == '':
            flash('No se seleccionó ningún archivo', 'danger')
            return redirect(request.url)
        
        if not target_unit:
            flash('No se seleccionó la unidad a entrenar (U1, U2, U3, U4)', 'danger')
            return redirect(request.url)
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'training', filename)
            file.save(filepath)
            
            try:
                metrics, model_filename = run_gradual_training(
                    filepath, 
                    target_unit, 
                    app.config['MODELS_FOLDER']
                )
                flash(f'Modelo para {target_unit} entrenado y guardado como {model_filename}!', 'success')
                return render_template('train_results_gradual.html', 
                                       metrics=metrics, 
                                       model_filename=model_filename, 
                                       target_unit=target_unit)
            except Exception as e:
                flash(f'Error durante el entrenamiento gradual: {str(e)}', 'danger')
                return redirect(request.url)
    
    # Si es GET, muestra la página de subida
    return render_template('train_gradual.html')


@app.route('/predict_gradual', methods=['GET', 'POST'])
def predict_gradual():
    """
    Página para predecir la SIGUIENTE unidad (U1, U2, U3, o U4).
    Detecta automáticamente qué predecir basado en las notas faltantes.
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
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction', filename)
            file.save(filepath)
            
            try:
                results_df, results_filename, prediction_target = run_gradual_prediction(
                    filepath, 
                    app.config['MODELS_FOLDER'], 
                    os.path.join(app.config['UPLOAD_FOLDER'], 'prediction')
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

# --- Rutas de Ayuda (Descargas y Gráficos) ---

@app.route('/uploads/prediction/<filename>')
def uploaded_file(filename):
    """Permite descargar el archivo CSV de resultados de predicción."""
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], 'prediction'), 
                               filename, 
                               as_attachment=True)

@app.route('/static/img/<filename>')
def send_img(filename):
    """Sirve las imágenes de gráficos generadas."""
    return send_from_directory(app.config['STATIC_IMG_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
