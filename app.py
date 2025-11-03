import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pipelines.training_pipeline import run_training
from pipelines.prediction_pipeline import run_prediction
# Importamos las nuevas pipelines graduales
from pipelines.training_pipeline_gradual import run_gradual_training
from pipelines.prediction_pipeline_gradual import run_gradual_prediction

app = Flask(__name__)

# --- Configuración de Carpetas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER_TRAIN = os.path.join(BASE_DIR, 'uploads', 'training')
UPLOAD_FOLDER_PREDICT = os.path.join(BASE_DIR, 'uploads', 'prediction')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
STATIC_IMG_FOLDER = os.path.join(BASE_DIR, 'static', 'img')

app.config['UPLOAD_FOLDER_TRAIN'] = UPLOAD_FOLDER_TRAIN
app.config['UPLOAD_FOLDER_PREDICT'] = UPLOAD_FOLDER_PREDICT
app.config['MODELS_FOLDER'] = MODELS_FOLDER
app.config['STATIC_IMG_FOLDER'] = STATIC_IMG_FOLDER

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

# --- Rutas de Entrenamiento ---

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    """
    Ruta para entrenar el modelo de predicción de PROMEDIO FINAL.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER_TRAIN'], filename)
            file.save(filepath)
            
            try:
                # Ejecutar el pipeline de entrenamiento
                metrics, plots = run_training(filepath, app.config['MODELS_FOLDER'], app.config['STATIC_IMG_FOLDER'])
                return render_template('train_results.html', metrics=metrics, plots=plots)
            except Exception as e:
                return f"Error durante el entrenamiento: {str(e)}"
    
    # Si es GET, solo muestra la página de subida
    return render_template('train.html') # Asumiremos que existe un train.html

@app.route('/train_gradual', methods=['GET', 'POST'])
def train_gradual_model():
    """
    NUEVA RUTA DE ENTRENAMIENTO GRADUAL
    Entrena un modelo específico para una unidad (U1, U2, U3, U4)
    basado en los datos históricos.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # NUEVO: Obtenemos la unidad que el usuario desea entrenar
        target_unit = request.form.get('target_unit') 
        
        if file.filename == '' or not target_unit:
            return "Error: No se seleccionó archivo o unidad de destino.", 400
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER_TRAIN'], filename)
            file.save(filepath)
            
            try:
                # Ejecutamos el pipeline de entrenamiento gradual
                metrics, plots, model_filename = run_gradual_training(
                    filepath, 
                    target_unit, 
                    app.config['MODELS_FOLDER'], 
                    app.config['STATIC_IMG_FOLDER']
                )
                return render_template('train_results_gradual.html', 
                                       metrics=metrics, 
                                       plots=plots, 
                                       model_filename=model_filename, 
                                       target_unit=target_unit)
            except Exception as e:
                return f"Error durante el entrenamiento gradual: {str(e)}"
    
    # Si es GET, muestra la página de subida para entrenamiento gradual
    return render_template('train_gradual.html') # Asumiremos que existe un train_gradual.html

# --- Rutas de Predicción ---

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Ruta para predecir el PROMEDIO FINAL con el modelo principal.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER_PREDICT'], filename)
            file.save(filepath)
            
            try:
                # Ejecutar el pipeline de predicción
                results_df, results_path = run_prediction(filepath, app.config['MODELS_FOLDER'], app.config['UPLOAD_FOLDER_PREDICT'])
                return render_template('prediction_results.html', 
                                       tables=[results_df.to_html(classes='data', header="true", index=False)], 
                                       results_path=results_path)
            except Exception as e:
                return f"Error durante la predicción: {str(e)}"
    
    # Si es GET, solo muestra la página de subida
    return render_template('predict.html') # Asumiremos que existe un predict.html

@app.route('/predict_gradual', methods=['GET', 'POST'])
def predict_gradual():
    """
    NUEVA RUTA DE PREDICCIÓN GRADUAL
    Detecta automáticamente qué unidad predecir (U1, U2, U3, U4)
    basado en las columnas de notas presentes en el archivo subido.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER_PREDICT'], filename)
            file.save(filepath)
            
            try:
                # Ejecutar el pipeline de predicción gradual "inteligente"
                results_df, results_path, prediction_target = run_gradual_prediction(
                    filepath, 
                    app.config['MODELS_FOLDER'], 
                    app.config['UPLOAD_FOLDER_PREDICT']
                )
                
                return render_template('prediction_results_gradual.html', 
                                       tables=[results_df.to_html(classes='data', header="true", index=False)], 
                                       results_path=results_path,
                                       prediction_target=prediction_target)
            except Exception as e:
                return f"Error durante la predicción gradual: {str(e)}"
    
    # Si es GET, solo muestra la página de subida
    return render_template('predict_gradual.html') # Asumiremos que existe predict_gradual.html

# --- Ruta para descargar resultados ---

@app.route('/uploads/prediction/<filename>')
def uploaded_file(filename):
    """Permite descargar el archivo de resultados de predicción."""
    return send_from_directory(app.config['UPLOAD_FOLDER_PREDICT'], filename)

if __name__ == '__main__':
    app.run(debug=True)
