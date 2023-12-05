import pandas as pd
import numpy as np
from functions.prediction_function import Predicting_Fertilizer
import plotly.graph_objects as go


def simulation_monte_carlo(
        temperature, humidity, ph, rainfall, label,
        min_percent, max_percent, number_simulation,
        model_n, model_p, model_k
):

    number_simulation = int(number_simulation)
    min_percent = min_percent / 100
    max_percent = max_percent / 100

    min_temperature = temperature - (temperature * min_percent)
    ideal_temperature = temperature
    max_temperature = temperature + (temperature * max_percent)

    min_humidity = humidity - (humidity * min_percent)
    ideal_humidity = humidity
    max_humidity = humidity + (humidity * max_percent)

    min_ph = ph - (ph * min_percent)
    ideal_ph = ph
    max_ph = ph + (ph * max_percent)

    min_rainfall = rainfall - (rainfall * min_percent)
    ideal_rainfall = rainfall
    max_rainfall = rainfall + (rainfall * max_percent)

    dist_temperature = np.random.normal(
        loc=ideal_temperature, 
        scale=(max_temperature - min_temperature) / 3, 
        size=number_simulation
        )
    dist_humidity = np.random.normal(
        loc=ideal_humidity, 
        scale=(max_humidity - min_humidity) / 3, 
        size=number_simulation
        )
    dist_ph = np.random.normal(
        loc=ideal_ph, 
        scale=(max_ph - min_ph) / 3, 
        size=number_simulation
        )
    dist_rainfall = np.random.normal(
        loc=ideal_rainfall, 
        scale=(max_rainfall - min_rainfall) / 3, 
        size=number_simulation
        )

    simulation_n = list()
    simulation_p = list()
    simulation_k = list()

    for i in range(0, number_simulation):
        get_temperature = np.random.choice(dist_temperature)
        get_humidity = np.random.choice(dist_humidity)
        get_ph = np.random.choice(dist_ph)
        get_rainfall = np.random.choice(dist_rainfall)

        predictor = Predicting_Fertilizer(get_temperature, get_humidity, get_ph, get_rainfall, label)
        predictor.predict_p(model_p)
        predictor.predict_n(model_n)
        predictor.predict_k(model_k)

        simulation_p.append(predictor.result_p)
        simulation_n.append(predictor.result_n)
        simulation_k.append(predictor.result_k)

    df_output = pd.DataFrame()
    df_output["N"] = simulation_n
    df_output["P"] = simulation_p
    df_output["K"] = simulation_k

    return df_output
