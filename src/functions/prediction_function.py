import pandas as pd
import pickle as pkl
import numpy as np

from pathlib import Path
import os


class Predicting_Fertilizer:
    def __init__(
            self, temperature, 
            humidity, ph, rainfall, label:str
            ) -> None:
        self.temperature = temperature
        self.humidity = humidity
        self.ph = ph
        self.rainfall = rainfall
        self.label = label

        __ROOT_PATH__ = Path(__file__).resolve().parent
        self.root_path = os.path.join(__ROOT_PATH__, "model")

        self.transform_log()

    def transform_log(self):
        self.temperature_log = np.log(self.temperature)
        self.humidity_log = np.log(self.humidity)
        self.rainfall_log = np.log(self.rainfall)
        self.ph_log = np.log(self.ph)

        labels = {
            'rice': 1, 'maize': 2, 'chickpea': 3,'kidneybeans': 4,
            'pigeonpeas': 5, 'mothbeans': 6, 'mungbean': 7, 'blackgram': 8,
            'lentil': 9,'pomegranate': 10, 'banana': 11, 'mango': 12,
            'grapes': 13, 'watermelon': 14, 'muskmelon': 15, 'apple': 16,
            'orange': 17, 'papaya': 18, 'coconut': 19, 'cotton': 20, 
            'jute': 21, 'coffee': 22
        }

        self.label_int = labels[self.label]


    def predict_p(self, model_name:str):
        with(open(os.path.join(self.root_path, model_name), "rb")) as model:
            model = pkl.load(model)

        cols = ["temperature", "humidity", "ph", "rainfall", "label"]
        data=[
                [
                    self.temperature_log, self.humidity_log, self.ph_log, 
                    self.rainfall_log, self.label_int
                ]
            ]
        X = pd.DataFrame(data=data, columns=cols)
        self.result_p = model.predict(X)[0]


    def predict_n(self, model_name:str):
        with(open(os.path.join(self.root_path, model_name), "rb")) as model:
            model = pkl.load(model)

        cols = ["P", "temperature", "humidity", "ph", "rainfall", "label"]
        data=[
                [
                    self.result_p, self.temperature_log, self.humidity_log, self.ph_log, 
                    self.rainfall_log, self.label_int
                ]
            ]
        X = pd.DataFrame(data=data, columns=cols)
        self.result_n = model.predict(X)[0]


    def predict_k(self, model_name:str):
        with(open(os.path.join(self.root_path, model_name), "rb")) as model:
            model = pkl.load(model)

        cols = ["N", "P", "temperature", "humidity", "ph", "rainfall", "label"]
        data=[
                [
                    self.result_n, self.result_p, self.temperature_log, 
                    self.humidity_log, self.ph_log, self.rainfall_log, self.label_int
                ]
            ]
        X = pd.DataFrame(data=data, columns=cols)
        self.result_k = model.predict(X)[0]


# Modelo não está prevendo modificações, o valor alterado chega no modelo, mas ele não prevê diferente. Checar o motivo
# Abrir um dos modelos em um notebook
# Pegar dados e transforma-los em log
# prever alguns dados para checar o motivo. Pode ser o arquivo pickle.
