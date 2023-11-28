import streamlit as st
import plotly.graph_objects as go
from functions.getters import Farfetch_Information
from pathlib import Path
import pandas as pd
import numpy as np
import os

from functions.prediction_function import Predicting_Fertilizer


__ROOT_PATH__ = Path().resolve().parent
file = "Crop_recommendation.csv"
data_path = os.path.join(__ROOT_PATH__, "data", file)

st.set_page_config(layout="wide")

## Creating Functions Here
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

####


st.title("NPK Recommendation")

df_information = Farfetch_Information(data_path)

col_1, col_2, col_3, col_4, col_5 = st.columns([5, 1, 2, 2, 2])

label = col_1.selectbox(
    label="Crop Type",
    options=df_information.labels
    )

temperature = col_1.number_input(
    label="Temperature", 
    min_value=-273.00,
    value=df_information.temperature,
    step=1.0
)
humidity = col_1.number_input(
    label="Humidity",
    value=df_information.humidity,
    step=1.0
)

ph = col_1.number_input(
    label="PH",
    value=df_information.ph,
    step=1.0
)
rainfall = col_1.number_input(
    label="Rainfall",
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
col_3.header(result_n)

col_4.markdown("## P")
col_4.header(result_p)


col_5.markdown("## K")
col_5.header(result_k)