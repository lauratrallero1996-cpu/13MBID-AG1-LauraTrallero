import json
import os


def test_model_performance_is_acceptable():
    """
    Test de regresión de modelo:
    - Comprueba que existe el archivo de métricas.
    - Verifica que f1 y recall son al menos 0.50
    """

    assert os.path.exists("models/metrics.json"), "No existe models/metrics.json"

    with open("models/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    f1 = metrics.get("f1")
    recall = metrics.get("recall")

    assert f1 is not None, "La métrica f1 no está en metrics.json"
    assert recall is not None, "La métrica recall no está en metrics.json"

    # Umbrales mínimos (ligeramente por debajo de lo que te ha salido)
    assert f1 >= 0.50, f"F1 demasiado baja: {f1}"
    assert recall >= 0.50, f"Recall demasiado baja: {recall}"
