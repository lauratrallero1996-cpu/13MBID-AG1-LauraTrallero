from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

TRAIN_FEATURE_COLUMNS = [
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

STREAMLIT_TO_TRAIN_COLS = {
    "age": "age",
    "campaign": "campaign",
    "previous": "previous",
    "duration": "duration",
    "emp_var_rate": "emp.var.rate",
    "cons_price_idx": "cons.price.idx",
    "cons_conf_idx": "cons.conf.idx",
    "euribor3m": "euribor3m",
    "nr_employed": "nr.employed",
}

app = FastAPI(title="Bank Marketing API")

model = joblib.load("models/model.pkl")


class ClientFeatures(BaseModel):
    age: int
    campaign: int
    previous: int
    duration: int
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float


@app.post("/predict")
def predict(features: ClientFeatures):
    try:
        data_in = features.dict()

        row = {}
        for streamlit_name, train_name in STREAMLIT_TO_TRAIN_COLS.items():
            row[train_name] = data_in[streamlit_name]

        row_ordered = {col: row[col] for col in TRAIN_FEATURE_COLUMNS}
        df = pd.DataFrame([row_ordered])

        y_pred = model.predict(df)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(df)[0][1])

        return {
            "prediction": int(y_pred),
            "probability_yes": proba,
        }

    except Exception as e:
        print("ERROR EN /predict:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
