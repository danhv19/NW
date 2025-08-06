
import os, joblib, warnings
import matplotlib
matplotlib.use("Agg")                      # backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from pipelines.preprocessing import load_and_clean_data

# Ajustes globales
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({"figure.figsize": (12, 7),
                     "axes.titlesize": 16,
                     "axes.labelsize": 12})


# ════════════════════════════════════════════════════════════════════
def run_training(file_path: str):
    # ─── FASE 1 · Carga y limpieza ───────────────────────────────────
    df = load_and_clean_data(file_path, is_training_data=True)

    if "ESTADO_APROBACION" not in df and "ESTADO_APROBACION_NUM" in df:
        df["ESTADO_APROBACION"] = df["ESTADO_APROBACION_NUM"].map({0: "0", 1: "1"})

    if "CUOTAS_PENDIENTES" not in df:
        cuota_cols = [c for c in df.columns if c.startswith("CUOTA_")]
        if cuota_cols:
            df["CUOTAS_PENDIENTES"] = df[cuota_cols].isnull().sum(axis=1)
            df["CUOTAS_PAGADAS"]    = 5 - df["CUOTAS_PENDIENTES"]

    cols = set(df.columns)

    # Carpetas
    os.makedirs("static/img", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    plot_files: list[str] = []

    def _save(fig, name: str):
        p = f"static/img/{name}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        plot_files.append(p.replace("static/", ""))

    # ─── EDA 1-3 (básicos) ──────────────────────────────────────────
    if {"ESTADO_APROBACION", "PORCENTAJE_asistencia"} <= cols:
        f, ax = plt.subplots()
        sns.boxplot(x="ESTADO_APROBACION", y="PORCENTAJE_asistencia",
                    data=df, ax=ax, palette={"0": "#FF6347", "1": "#90EE90"})
        ax.set_title("Asistencia vs. Aprobación")
        _save(f, "eda_box_asistencia")

    if {"CUOTAS_PENDIENTES", "ESTADO_APROBACION"} <= cols:
        f, ax = plt.subplots()
        sns.countplot(x="CUOTAS_PENDIENTES", hue="ESTADO_APROBACION",
                      data=df, ax=ax, palette={"0": "#FF6347", "1": "#90EE90"})
        ax.set_title("Cuotas pendientes vs. Resultado")
        _save(f, "eda_count_cuotas")

    if {"ESTADO_APROBACION", "EDAD"} <= cols:
        f, ax = plt.subplots()
        sns.boxplot(x="ESTADO_APROBACION", y="EDAD", data=df, ax=ax)
        ax.set_title("Edad vs. Resultado")
        _save(f, "eda_box_edad")

    # ─── EDA 4 KDE % asistencia ─────────────────────────────────────
    if {"PORCENTAJE_asistencia", "ESTADO_APROBACION"} <= cols:
        f, ax = plt.subplots()
        sns.kdeplot(data=df, x="PORCENTAJE_asistencia", hue="ESTADO_APROBACION",
                    fill=True, common_norm=False,
                    palette={"0": "#FF6347", "1": "#90EE90"}, ax=ax)
        ax.set_title("Curvas KDE • % Asistencia")
        _save(f, "eda_kde_asistencia")

    # ─── EDA 5 Probabilidad aprobar vs % asistencia ────────────────
    if {"PORCENTAJE_asistencia", "ESTADO_APROBACION_NUM"} <= cols:
        X_plot = df["PORCENTAJE_asistencia"].values.reshape(-1, 1)
        y_plot = df["ESTADO_APROBACION_NUM"].values
        logreg = LogisticRegression(max_iter=500).fit(X_plot, y_plot)

        xs = np.linspace(X_plot.min(), X_plot.max(), 300).reshape(-1, 1)
        ys = logreg.predict_proba(xs)[:, 1]

        f, ax = plt.subplots()
        ax.scatter(X_plot, y_plot, alpha=.3, s=25, label="Datos")
        ax.plot(xs, ys, color="blue", lw=2, label="Curva logística")
        ax.set_xlabel("% Asistencia")
        ax.set_ylabel("Probabilidad de aprobar")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Probabilidad de aprobar vs. % Asistencia")
        ax.legend()
        _save(f, "eda_reglog_prob_aprobar")

    # ─── EDA 6-9 adicionales ────────────────────────────────────────
    if {"CUOTAS_PENDIENTES", "PROMEDIO_CURSO"} <= cols:
        f, ax = plt.subplots()
        sns.boxplot(x="CUOTAS_PENDIENTES", y="PROMEDIO_CURSO", data=df, ax=ax)
        ax.set_title("Promedio por Nº de Cuotas pendientes")
        _save(f, "eda_box_cuotas_promedio")

    if {"DESCRIPCION", "PROMEDIO_CURSO"} <= cols:
        top_car = df["DESCRIPCION"].value_counts().nlargest(10).index
        f, ax = plt.subplots(figsize=(10,5))
        sns.boxplot(x="DESCRIPCION", y="PROMEDIO_CURSO",
                    data=df[df["DESCRIPCION"].isin(top_car)], ax=ax)
        ax.set_title("Promedio por Carrera (Top 10)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        _save(f, "eda_box_promedio_carrera")

    if {"DOCENTE", "PROMEDIO_CURSO"} <= cols:
        d = (df.groupby("DOCENTE")["PROMEDIO_CURSO"]
               .agg(["mean", "count"])
               .query("count >= 10")
               .nlargest(20, "mean")
               .reset_index())
        f, ax = plt.subplots(figsize=(8,10))
        sns.barplot(x="mean", y="DOCENTE", data=d, palette="viridis", ax=ax)
        ax.set_title("Promedio medio por Docente (≥10 alumnos)")
        _save(f, "eda_bar_promedio_docente")

    if {"PROCEDENCIA", "PROMEDIO_CURSO"} <= cols:
        top_proc = df["PROCEDENCIA"].value_counts().nlargest(15).index
        f, ax = plt.subplots(figsize=(10,5))
        sns.boxplot(x="PROCEDENCIA", y="PROMEDIO_CURSO",
                    data=df[df["PROCEDENCIA"].isin(top_proc)], ax=ax)
        ax.set_title("Promedio por Procedencia (Top 15)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        _save(f, "eda_box_promedio_procedencia")

    # ─── EDA 10-12 Edad, cursos, cuotas pagadas ─────────────────────
    if {"EDAD", "PROMEDIO_CURSO"} <= cols:
        f, ax = plt.subplots()
        sns.regplot(x="EDAD", y="PROMEDIO_CURSO", data=df,
                    scatter_kws={"alpha": .4}, line_kws={"color": "red"}, ax=ax)
        ax.set_title("Edad vs. Promedio del curso")
        _save(f, "eda_reg_edad_promedio")

    if {"CURSOS_MATRICULADOS", "PROMEDIO_CURSO"} <= cols:
        f, ax = plt.subplots()
        sns.boxplot(x="CURSOS_MATRICULADOS", y="PROMEDIO_CURSO", data=df, ax=ax)
        ax.set_title("Promedio por Nº de cursos matriculados")
        _save(f, "eda_box_promedio_cursos")

    if {"CUOTAS_PAGADAS", "PROMEDIO_CURSO"} <= cols:
        f, ax = plt.subplots()
        sns.boxplot(x="CUOTAS_PAGADAS", y="PROMEDIO_CURSO", data=df, ax=ax)
        ax.set_title("Promedio por Nº de Cuotas pagadas")
        _save(f, "eda_box_promedio_cuotas_pag")

    # ─── EDA 13 Matriz de correlación ───────────────────────────────
    num_cols_all = df.select_dtypes("number").columns
    if len(num_cols_all) >= 2:
        corr = df[num_cols_all].corr()
        f, ax = plt.subplots(figsize=(11,9))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Matriz de correlación (variables numéricas)")
        _save(f, "eda_heatmap_corr")

    # ─── FASE 2 · Preparación de datos ───────────────────────────────
    drop_cols = [
        "Periodo","CÓDIGO","ALUMNO","DEPART","DISTRITO","NACIM","PERIODO_INGRESO",
        "CARRERA_PLAN","FECHA_MATRICULA","CURSO_CODIGO","CURSO_NOMBRE","DESCRIPCION_1",
        "PROMEDIO_CURSO","TIPOSESION","SECCIÓN","DOCENTE","PERIODO_ACADEMICO","PERIODOMES",
        "ASISTENCIAS","CLASES","Promedio_Final","CUOTA_1","CUOTA_2","CUOTA_3","CUOTA_4","CUOTA_5",
        "ESTADO_APROBACION","DESCRIPCION"
    ]
    X = df.drop(columns=drop_cols + ["ESTADO_APROBACION_NUM"], errors="ignore")
    y = df["ESTADO_APROBACION_NUM"]

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    X[cat_cols] = X[cat_cols].astype(str)

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh",  OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=.25, stratify=y, random_state=42)

    # ─── Entrenamiento y selección del mejor modelo ─────────────────
    models = {
        "LogReg": (
            LogisticRegression(class_weight="balanced", random_state=42),
            {"classifier__C":[.1,1,10], "classifier__solver":["liblinear"]}
        ),
        "RandForest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {"classifier__n_estimators":[100,200], "classifier__max_depth":[10,None]}
        ),
        "GradBoost": (
            GradientBoostingClassifier(random_state=42),
            {"classifier__n_estimators":[100,200],
             "classifier__learning_rate":[.05,.1],
             "classifier__max_depth":[3,5]}
        )
    }

    best_pipe, best_name, best_acc = None, "", 0.0
    logs = []

    for name, (clf, grid) in models.items():
        pipe = Pipeline([("preprocessor", pre), ("classifier", clf)])
        gs   = GridSearchCV(pipe, grid, cv=5, n_jobs=-1,
                            scoring="roc_auc").fit(X_tr, y_tr)

        y_pred = gs.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        logs.append(f"{name}: {acc:.4f} | {gs.best_params_}")

        # matriz de confusión
        cm = confusion_matrix(y_te, y_pred)
        f, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Desap.","Aprob."],
                    yticklabels=["Desap.","Aprob."], ax=ax)
        ax.set_title(f"Matriz de confusión – {name}")
        _save(f, f"conf_{name.lower()}")

        if acc > best_acc:
            best_pipe, best_name, best_acc = gs.best_estimator_, name, acc

    # ─── Guardar el mejor modelo ────────────────────────────────────
    model_file = f"modelo_{best_name.lower()}.pkl"
    joblib.dump(best_pipe, os.path.join("models", model_file))

    summary = "\n".join(logs) + f"\n\n🏆 MEJOR MODELO → {best_name} (ACC {best_acc:.4f})"
    return summary, model_file, plot_files
