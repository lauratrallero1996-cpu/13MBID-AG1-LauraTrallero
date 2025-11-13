import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Ruta al CSV original (ajusta si tu estructura es diferente)
DATA_PATH = Path("data/raw/bank-additional-full.csv")
OUTPUT_DIR = Path("reports/figures")

def main():
    # Crear carpeta de salida si no existe
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar datos (usa el mismo separador que en el notebook)
    df = pd.read_csv(DATA_PATH, sep=';')

    # 1) Histograma de edad
    plt.figure()
    df['age'].hist(bins=20)
    plt.title("Distribución de la edad")
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_age_hist.png")
    plt.close()

    # 2) Distribución de la variable objetivo y
    plt.figure()
    df['y'].value_counts().plot(kind='bar')
    plt.title("Distribución de la variable objetivo (y)")
    plt.xlabel("Clase")
    plt.ylabel("Número de registros")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_target_y_bar.png")
    plt.close()

    # 3) Ejemplo extra: top 10 profesiones (si existe la columna 'job')
    if 'job' in df.columns:
        plt.figure()
        df['job'].value_counts().head(10).plot(kind='barh')
        plt.title("Top 10 profesiones")
        plt.xlabel("Número de registros")
        plt.ylabel("Profesión")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "03_top10_job.png")
        plt.close()

if __name__ == "__main__":
    main()
