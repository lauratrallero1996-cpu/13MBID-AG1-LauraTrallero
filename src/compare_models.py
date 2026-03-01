import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def load_data():
    df = pd.read_csv(
        "data/processed/bank-additional-processed.csv",
        sep=";"
    )

    # columna objetivo
    y_raw = df["y"]
    if y_raw.dtype == object:
        y = (y_raw.str.lower() == "yes").astype(int)
    else:
        y = y_raw.astype(int)

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

    X = df[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def eval_model(name, clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = {
        "model": name,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, pos_label=1),
        "test_recall": recall_score(y_test, y_pred, pos_label=1),
        "test_f1": f1_score(y_test, y_pred, pos_label=1),
    }
    return metrics


def main():
    X_train, X_test, y_train, y_test = load_data()

    models = [
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, solver="liblinear", random_state=1)),
        ("Linear SVC",
         LinearSVC(C=1.0, random_state=1)),
        ("KNN (k=5)",
         KNeighborsClassifier(n_neighbors=5)),
        ("Decision Tree",
         DecisionTreeClassifier(criterion="gini", random_state=1)),
    ]

    results = []
    for name, clf in models:
        print("=" * 60)
        print(f"Modelo: {name}")
        metrics = eval_model(name, clf, X_train, X_test, y_train, y_test)
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
        results.append(metrics)

    # opcional: guardar a JSON para consultarlo con calma
    with open("models/compare_models_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

