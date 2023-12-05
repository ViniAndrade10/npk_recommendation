import streamlit as st
import plotly.graph_objects as go
from functions.getters import Farfetch_Information
from pathlib import Path
import pandas as pd
import numpy as np
import os

from functions.prediction_function import Predicting_Fertilizer
from functions.monte_carlo_simulation import simulation_monte_carlo
from functions.make_plots import plot_histogram


__ROOT_PATH__ = Path().resolve().parent
file = "Crop_recommendation.csv"
data_path = os.path.join(__ROOT_PATH__, "data", file)

st.set_page_config(
    layout="wide",
    # page_icon=":smiley:",
    page_title="Seu Aplicativo"
    )

style_sheet = "../src/styles/styles.css"
with open(style_sheet) as f:
    st.markdown(f"<style>{f.read}</style>", unsafe_allow_html=True)

## Creating Functions Here
@st.cache_data()
def predict_values(
        temperature, humidity, ph, rainfall, label,
        model_n, model_p, model_k
):
    predictor = Predicting_Fertilizer(temperature, humidity, ph, rainfall, label)
    predictor.predict_p(model_p)
    predictor.predict_n(model_n)
    predictor.predict_k(model_k)

    predict_p = predictor.result_p
    predict_n = predictor.result_n
    predict_k = predictor.result_k

    return predict_n, predict_p, predict_k

@st.cache_data()
def plot_hist(df_simulation, column, number_simulation):
    fig = plot_histogram(df_simulation, column, number_simulation)
    return fig


st.title("NPK Recommendation")

df_information = Farfetch_Information(data_path)

col_1, col_2, col_3, col_4, col_5 = st.columns([5, 1, 2, 2, 2])

label = col_1.selectbox(
    label="Crop Type",
    options=df_information.labels
    )

temperature = col_1.number_input(
    label="Temperature (ºC)", 
    min_value=df_information.temperature_min,
    max_value=df_information.temperature_max,
    value=round(df_information.temperature, ndigits=0),
    step=1.0
)
humidity = col_1.number_input(
    label="Humidity (%)",
    min_value=df_information.humidity_min,
    max_value=df_information.humidity_max,
    value=df_information.humidity,
    step=1.0
)

ph = col_1.number_input(
    label="pH",
    min_value=df_information.ph_min,
    max_value=df_information.ph_max,
    value=round(df_information.ph, ndigits=1),
    step=1.0
)
rainfall = col_1.number_input(
    label="Rainfall (mm / Month)",
    min_value=df_information.rainfall_min,
    max_value=df_information.rainfall_max,
    value=df_information.rainfall,
    step=1.0
)

model_path = os.path.join(__ROOT_PATH__, "src", "model")
model_n = os.path.join(model_path, "model_recommentation_n.pkl")
model_p = os.path.join(model_path, "model_recommentation_p.pkl")
model_k = os.path.join(model_path, "model_recommentation_k.pkl")


result_n, result_p, result_k = predict_values(
    temperature, humidity, ph, rainfall, 
    label, model_n, model_p, model_k
)

col_3.markdown("## N")
col_3.header(result_n.round(3))

col_4.markdown("## P")
col_4.header(result_p.round(3))

col_5.markdown("## K")
col_5.header(result_k.round(3))

with st.expander("Simulate Scenarios"):
    col_expander_1, col_expander_2, col_expander_3 = st.columns([2, 1, 9])
    min_percent = col_expander_1.number_input(
        label="% Minimum - Simulation",
        min_value=0.10,
        max_value=100.00,
        step=10.00
    )
    max_percent = col_expander_1.number_input(
        label="% Maximum - Simulation",
        min_value=0.10,
        max_value=100.00,
        step=10.00
    )
    number_simulation = col_expander_1.number_input(
        label="Number of Simulations",
        min_value=100.00,
        max_value=10000.00,
        value=1000.00,
        step=100.00
    )

    button_execute =  col_expander_1.button("Execute Simulation")

    if button_execute:
        df_simulation = simulation_monte_carlo(
            temperature, humidity, ph, rainfall, label,
            min_percent, max_percent, number_simulation,
            model_n, model_p, model_k
        )

        categories = [
            "Temperature", "Humidity", "pH", "Rainfall"
        ]

        tab_1, tab_2, tab_3 = col_expander_3.tabs(["N", "P", "K"])

        tab_1.plotly_chart(plot_hist(df_simulation, "N", number_simulation))

        tab_2.plotly_chart(plot_hist(df_simulation, "P", number_simulation))

        tab_3.plotly_chart(plot_hist(df_simulation, "K", number_simulation))
