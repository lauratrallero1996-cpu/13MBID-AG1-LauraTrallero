import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/bank-additional-full.csv")
OUTPUT_PATH = Path("data/processed/bank-additional-processed.csv")

def load_raw():
    return pd.read_csv(RAW_PATH, sep=';')

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # --- 1) Eliminar duplicados ---
    df = df.drop_duplicates()

    # --- 2) Reemplazar unknown por NaN ---
    df = df.replace("unknown", pd.NA)

    # --- 3) Eliminar filas con NaN ---
    df = df.dropna()

    # --- 4) Eliminar columna 'default' si está llena de unknowns o no aporta ---
    if 'default' in df.columns:
        df = df.drop(columns=['default'])

    # --- 5) Crear variable binaria según pdays (si existe) ---
    if 'pdays' in df.columns:
        df['was_contacted_before'] = (df['pdays'] != 999).astype(int)
        df = df.drop(columns=['pdays'])

    # --- 6) Convertir la variable objetivo a binaria si hace falta ---
    if 'y' in df.columns:
        df['y'] = df['y'].map({'yes': 1, 'no': 0})

    # --- 7) Reset index ---
    df = df.reset_index(drop=True)

    return df

def main():
    df = load_raw()
    df_clean = prepare(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False, sep=';')

    print(f"Archivo procesado guardado en: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
