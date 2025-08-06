# ───────────────────────────── app.py ─────────────────────────────
from flask import Flask, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename
import os
from pipelines.training_pipeline import run_training
from pipelines.prediction_pipeline import run_prediction

# ── Configuración ─────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"xlsx", "xls", "pkl"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MODEL_FOLDER"] = "models"

# ── Utilidades ────────────────────────────────────────────────────
def allowed_file(filename: str, allowed: set) -> bool:
    """Valida únicamente la extensión del archivo."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


# ── Rutas ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ---------- Entrenamiento -----------------------------------------------------
@app.route("/train", methods=["POST"])
def train():
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


# ---------- Predicción --------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data_f = request.files.get("predict_file")
    model_f = request.files.get("model_file")

    if (
        data_f
        and allowed_file(data_f.filename, {"xlsx", "xls"})
        and model_f
        and allowed_file(model_f.filename, {"pkl"})
    ):
        # Guardar Excel de datos nuevos
        pred_dir = os.path.join(app.config["UPLOAD_FOLDER"], "prediction")
        os.makedirs(pred_dir, exist_ok=True)
        data_name = secure_filename(data_f.filename)
        data_path = os.path.join(pred_dir, data_name)
        data_f.save(data_path)

        # Guardar modelo .pkl
        os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)
        model_name = secure_filename(model_f.filename)
        model_path = os.path.join(app.config["MODEL_FOLDER"], model_name)
        model_f.save(model_path)

        # Ejecutar pipeline de predicción
        df_out, excel_path = run_prediction(data_path, model_path)
        table_html = df_out.to_html(classes="table table-striped table-hover", index=False)

        return render_template(
            "prediction_results.html",
            table=table_html,
            pred_file=os.path.basename(excel_path),
        )

    return "Archivos inválidos o faltantes.", 400


# ---------- Descargas ---------------------------------------------------------
@app.route("/download/prediction/<filename>")
def download_prediction_file(filename: str):
    return send_file(
        os.path.join("uploads", "prediction", filename),
        as_attachment=True,
    )


@app.route("/download/model/<filename>")
def download_model_file(filename: str):
    return send_file(
        os.path.join("models", filename),
        as_attachment=True,
    )


# ── Inicialización de carpetas y servidor ─────────────────────────────────────
if __name__ == "__main__":
    # Asegurar directorios base
    for d in ["uploads/training", "uploads/prediction", "models", "static/img"]:
        os.makedirs(d, exist_ok=True)

    app.run(debug=True)
