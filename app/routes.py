from app import app
from flask import render_template, request
from src.functions.getters import Farfetch_Information
from src.functions.prediction_function import Predicting_Fertilizer

from pathlib import Path
import pandas as pd
import numpy as np
import os


__ROOT_PATH__ = Path().resolve()
file = "Crop_recommendation.csv"
data_path = os.path.join(__ROOT_PATH__, "data", file)

model_path = os.path.join(__ROOT_PATH__, "src", "model")
model_n = os.path.join(model_path, "model_recommentation_n.pkl")
model_p = os.path.join(model_path, "model_recommentation_p.pkl")
model_k = os.path.join(model_path, "model_recommentation_k.pkl")


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/recommendation-model",  methods=["POST", "GET"])
# @app.route("/recommendation-model", defaults={"temperature":0, "humidity":0, "ph":0, "rainfall":0, "label":"Rice"})
def recommendation():
    df_info = Farfetch_Information(data_path)

    avg_temperature = round(df_info.temperature, ndigits=1)
    avg_humidity = round(df_info.humidity, ndigits=1)
    avg_ph = round(df_info.ph, ndigits=1)
    avg_rainfall = round(df_info.rainfall, ndigits=1)
    std_label = df_info.labels[0]

    temperature = request.form.get("temperature", default=avg_temperature)
    humidity = request.form.get("humidity", default=avg_humidity)
    ph = request.form.get("ph", default=avg_ph)
    rainfall = request.form.get("rainfall", default=avg_rainfall)
    label = request.form.get("label", default=std_label)

    if temperature is None or temperature == '':
        temperature = avg_temperature
    if humidity is None or humidity == '':
        humidity = avg_humidity
    if ph is None or ph == '':
        ph = avg_ph
    if rainfall is None or rainfall == '':
        rainfall = avg_rainfall
    if label is None or label == '':
        label = std_label

    data_post = {
        "temperature":temperature,
        "humidity":humidity,
        "ph":ph,
        "rainfall":rainfall,
        "label":label
    }
    print(data_post)

    predictor = Predicting_Fertilizer(
        float(temperature), 
        float(humidity), 
        float(ph), 
        float(rainfall), 
        label.lower()
        )
    predictor.predict_p(model_p)
    predictor.predict_n(model_n)
    predictor.predict_k(model_k)

    predict_p = predictor.result_p
    predict_n = predictor.result_n
    predict_k = predictor.result_k

    return render_template(
        "recommendation.html", 
        labels=df_info.labels,
        data_post=data_post,
        temperature=temperature,
        humidity=humidity, 
        ph=ph, rainfall=rainfall, label=label.title(),
        predict_p=round(predict_p, ndigits=2),
        predict_n=round(predict_n, ndigits=2),
        predict_k=round(predict_k, ndigits=2)
        )


@app.route("/model")
def model():
    return render_template("model.html")
