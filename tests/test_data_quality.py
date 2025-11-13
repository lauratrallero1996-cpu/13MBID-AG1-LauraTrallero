import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/bank-additional-full.csv")

def load_data():
    return pd.read_csv(DATA_PATH, sep=';')

# 1. No duplicados
def test_no_duplicates():
    df = load_data()
    total_rows = len(df)
    unique_rows = len(df.drop_duplicates())
    duplicate_rows = total_rows - unique_rows
    duplicate_ratio = duplicate_rows / total_rows

    # Aceptamos hasta un 5% de filas duplicadas
    assert duplicate_ratio < 0.05, (
        f"Demasiadas filas duplicadas: {duplicate_rows} "
        f"({duplicate_ratio:.4%} del total)"
    )
# 2. No valores nulos
def test_target_not_null():
    df = load_data()
    assert df['y'].isna().sum() == 0, "La variable y tiene valores nulos"

# 3. Rango de edad tolerable (entre 17 y 100 años, por ejemplo)
def test_age_range():
    df = load_data()
    # En este dataset real las edades van de 17 a 98
    assert df['age'].between(17, 100).all(), "La columna age tiene valores fuera de rango razonable"

# 4. Proporción de unknown <80%
def test_unknown_ratio():
    df = load_data()

    cols = [c for c in df.columns if c not in ['y', 'age']]

    for col in cols:
        if df[col].dtype == object:
            prop_unknown = (df[col] == "unknown").mean()
            assert prop_unknown < 0.8, f"Demasiados 'unknown' en {col}: {prop_unknown:.2f}"

# 5. Sin nulos en columnas criticas
def test_no_nulls_in_critical_columns():
    df = load_data()
    critical_cols = ['age', 'job', 'marital', 'education', 'y']
    for col in critical_cols:
        assert df[col].isna().sum() == 0, f"La columna {col} tiene nulos"
