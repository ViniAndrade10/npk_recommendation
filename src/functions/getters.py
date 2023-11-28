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


    def get_humidity(self):
        self.humidity = self.df["humidity"].mean()


    def get_ph(self):
        self.ph = self.df["ph"].mean()


    def get_rainfall(self):
        self.rainfall = self.df["rainfall"].mean()


    def get_labels(self):
        self.labels = self.df["label"].unique()


# Make getter to take maximum value and minimum values for each input number