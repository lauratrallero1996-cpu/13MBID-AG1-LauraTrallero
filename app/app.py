import streamlit as st
import requests

st.title("Predicción de apertura de depósito a plazo")

st.write("Introduce los datos del cliente y la campaña:")

age = st.number_input("Edad", min_value=18, max_value=100, value=40)
campaign = st.number_input("Número de contactos en esta campaña", min_value=1, value=1)
previous = st.number_input("Número de contactos previos", min_value=0, value=0)
duration = st.number_input("Duración de la llamada (segundos)", min_value=0, value=100)
emp_var_rate = st.number_input("Tasa de variación del empleo", value=1.1)
cons_price_idx = st.number_input("Índice de precios al consumidor", value=93.0)
cons_conf_idx = st.number_input("Índice de confianza del consumidor", value=-40.0)
euribor3m = st.number_input("Euribor 3 meses", value=4.0)
nr_employed = st.number_input("Número de empleados", value=5000.0)

if st.button("Predecir"):
    payload = {
        "age": age,
        "campaign": campaign,
        "previous": previous,
        "duration": duration,
        "emp_var_rate": emp_var_rate,
        "cons_price_idx": cons_price_idx,
        "cons_conf_idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr_employed": nr_employed,
    }

    try:
        resp = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Predicción (1 = abre depósito, 0 = no): {data['prediction']}")
            if data["probability_yes"] is not None:
                st.info(f"Probabilidad de que abra depósito: {data['probability_yes']:.2%}")
        else:
            st.error(f"Error al llamar a la API: {resp.status_code}")
    except Exception as e:
        st.error(f"No se pudo conectar con la API: {e}")
