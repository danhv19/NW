from flask import Flask, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename
import os
import pandas as pd
from pipelines.training_pipeline import run_training
from pipelines.prediction_pipeline import run_prediction
from pipelines.training_pipeline_gradual import train_and_save_gradual_models
from pipelines.prediction_pipeline_gradual import run_gradual_prediction

# --- Configuración de la Aplicación ---
ALLOWED_EXTENSIONS = {"xlsx", "xls", "pkl"}
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MODEL_FOLDER"] = "models"


# --- Funciones de Utilidad ---
def allowed_file(filename: str, allowed: set) -> bool:
    """Valida la extensión del archivo."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


# --- Rutas Principales ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    """Ruta para el entrenamiento de modelo original."""
    f = request.files.get("train_file")
    if f and allowed_file(f.filename, {"xlsx", "xls"}):
        safe_name = secure_filename(f.filename)
        train_dir = os.path.join(app.config["UPLOAD_FOLDER"], "training")
        os.makedirs(train_dir, exist_ok=True)
        path = os.path.join(train_dir, safe_name)
        f.save(path)

        summary, model_fname, plots = run_training(path)
        return render_template(
            "train_results.html",
            results=summary,
            model_name=model_fname,
            plots=plots,
        )
    return "Archivo de entrenamiento no válido.", 400

@app.route('/train_gradual', methods=['POST'])
def train_gradual_route():
    """Ruta para el entrenamiento de modelo gradual por etapas."""
    if 'train_file' not in request.files:
        return "No se encontró el archivo", 400
    
    file = request.files['train_file']
    if file.filename == '':
        return "No se seleccionó ningún archivo", 400

    if file and (file.filename.endswith('.xlsx')):
        filename = secure_filename(file.filename)
        train_data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'training', filename)
        file.save(train_data_path)
        
        training_results = train_and_save_gradual_models(train_data_path)
        
        image_paths_raw = training_results.pop('image_paths', {})
        
        # Limpia las rutas de las imágenes para que sean compatibles con la web
        images_cleaned = {
            title: path.replace('\\', '/').replace('static/', '') 
            for title, path in image_paths_raw.items()
        }
        
        return render_template('train_results_gradual.html', 
                               results=training_results,
                               images=images_cleaned)
    else:
        return "Formato de archivo no válido. Por favor, sube un .xlsx", 400

@app.route("/predict", methods=["POST"])
def predict():
    """Ruta para realizar predicciones con un modelo."""
    data_f = request.files.get("predict_file")
    model_f = request.files.get("model_file")

    if (
        data_f and allowed_file(data_f.filename, {"xlsx", "xls"}) and
        model_f and allowed_file(model_f.filename, {"pkl"})
    ):
        pred_dir = os.path.join(app.config["UPLOAD_FOLDER"], "prediction")
        os.makedirs(pred_dir, exist_ok=True)
        data_name = secure_filename(data_f.filename)
        data_path = os.path.join(pred_dir, data_name)
        data_f.save(data_path)

        os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)
        model_name = secure_filename(model_f.filename)
        model_path = os.path.join(app.config["MODEL_FOLDER"], model_name)
        model_f.save(model_path)

        df_out, excel_path = run_prediction(data_path, model_path)
        table_html = df_out.to_html(classes="table table-striped table-hover", index=False)

        return render_template(
            "prediction_results.html",
            table=table_html,
            pred_file=os.path.basename(excel_path),
        )

    return "Archivos inválidos o faltantes.", 400


# --- Rutas de Descarga ---
@app.route("/download/prediction/<filename>")
def download_prediction_file(filename: str):
    """Descarga el archivo de predicción resultante."""
    return send_file(
        os.path.join("uploads", "prediction", filename),
        as_attachment=True,
    )

@app.route("/download/model/<filename>")
def download_model_file(filename: str):
    """Descarga un archivo de modelo entrenado."""
    return send_file(
        os.path.join("models", filename),
        as_attachment=True,
    )




    # ---------- NUEVA RUTA PARA PREDICCIÓN GRADUAL -----------------------------
@app.route("/predict_gradual", methods=["POST"])
def predict_gradual_route():
    data_f = request.files.get("predict_file")
    model_f = request.files.get("model_file")

    if (
        data_f and allowed_file(data_f.filename, {"xlsx", "xls"}) and
        model_f and allowed_file(model_f.filename, {"pkl"})
    ):
        pred_dir = os.path.join(app.config["UPLOAD_FOLDER"], "prediction")
        os.makedirs(pred_dir, exist_ok=True)
        data_name = secure_filename(data_f.filename)
        data_path = os.path.join(pred_dir, data_name)
        data_f.save(data_path)

        os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)
        model_name = secure_filename(model_f.filename)
        model_path = os.path.join(app.config["MODEL_FOLDER"], model_name)
        model_f.save(model_path)

        df_out, error = run_gradual_prediction(data_path, model_path)
        
        if error:
            # Manejar el error, por ejemplo, mostrándolo en la página
            return f"Ocurrió un error: {error}", 500

        table_html = df_out.to_html(classes="table table-striped table-hover", index=False)

        return render_template(
            "prediction_results.html",
            table=table_html,
            pred_file=None # La descarga del excel se puede añadir después
        )

    return "Archivos inválidos o faltantes para la predicción gradual.", 400

# --- Inicialización del Servidor ---
if __name__ == "__main__":
    # Asegura la creación de los directorios necesarios al iniciar
    for d in ["uploads/training", "uploads/prediction", "models", "static/img"]:
        os.makedirs(d, exist_ok=True)

    app.run(debug=True)