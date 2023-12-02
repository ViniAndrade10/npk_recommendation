from pathlib import Path
import pandas as pd
import numpy as np
import os


__ROOT_PATH__ = Path().resolve()
file = "Crop_recommendation.csv"
data_path = os.path.join(__ROOT_PATH__, "data", "")

class Farfetch_Information:
    def __init__(self, path) -> None:
        self.path = path

        self.read_df_information()
        self.get_temperature()
        self.get_humidity()
        self.get_ph()
        self.get_rainfall()
        self.get_labels()


    def read_df_information(self):
        self.df = pd.read_csv(self.path, sep=",")


    def get_temperature(self):
        self.temperature = self.df["temperature"].mean()
        self.temperature_min = self.df["temperature"].min()
        self.temperature_max = self.df["temperature"].max()


    def get_humidity(self):
        self.humidity = self.df["humidity"].mean()
        self.humidity_min = self.df["humidity"].min()
        self.humidity_max = self.df["humidity"].max()


    def get_ph(self):
        self.ph = self.df["ph"].mean()
        self.ph_min = self.df["ph"].min()
        self.ph_max = self.df["ph"].max()


    def get_rainfall(self):
        self.rainfall = self.df["rainfall"].mean()
        self.rainfall_min = self.df["rainfall"].min()
        self.rainfall_max = self.df["rainfall"].max()


    def get_labels(self):
        self.labels = self.df["label"].unique()


# Make getter to take maximum value and minimum values for each input number