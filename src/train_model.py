import json
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    # =========================================================
    # 1. Cargar datos procesados
    # =========================================================
    df = pd.read_csv(
        "data/processed/bank-additional-processed.csv",
        sep=";"  # el fichero va separado por ;
    )

    # ---------------------------------------------------------
    # Detectar automáticamente la columna objetivo
    # ---------------------------------------------------------
    possible_targets = ["y", "y_yes", "target", "objetivo", "target_y"]
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError(
            f"No se ha encontrado la columna objetivo. Columnas disponibles: {list(df.columns)}"
        )

    print(f"Usando la columna objetivo: {target_col}")

    # =========================================================
    # 2. Separar X e y y convertir tipos
    # =========================================================
    # y: convertimos a 0/1 si viene como 'yes'/'no'
    y_raw = df[target_col]

    if y_raw.dtype == object:
        # normalizamos a minúsculas por si acaso
        y = (y_raw.str.lower() == "yes").astype(int)
    else:
        y = y_raw.astype(int)

    # X: SOLO las columnas que usará el MVP (API + Streamlit)
    feature_columns = [
        "age",
        "campaign",
        "previous",
        "duration",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]

    # comprobamos que existen todas las columnas requeridas
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {missing}")

    X = df[feature_columns].copy()

    # por seguridad, nos aseguramos de que todo es numérico
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # =========================================================
    # 3. Dividir datos
    # =========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # =========================================================
    # 4. Definir y entrenar modelo
    # =========================================================
    params = {
        "criterion": "gini",
        "max_depth": None,
        "random_state": 42,
    }
    clf = DecisionTreeClassifier(**params)

    # =========================================================
    # 5. Registrar con MLflow
    # =========================================================
    mlflow.set_experiment("bank_marketing_models")

    with mlflow.start_run(run_name="decision_tree_mvp"):
        mlflow.log_params(params)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label=1),
            "recall": recall_score(y_test, y_pred, pos_label=1),
            "f1": f1_score(y_test, y_pred, pos_label=1),
            "test_samples": int(len(y_test)),
            "train_samples": int(len(y_train)),
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        mlflow.sklearn.log_model(clf, "model")

        # =====================================================
        # 6. Guardar modelo y métricas en disco
        # =====================================================
        os.makedirs("models", exist_ok=True)

        joblib.dump(clf, "models/model.pkl")

        with open("models/metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print("Modelo y métricas guardados correctamente.")
        print(metrics)


if __name__ == "__main__":
    main()

