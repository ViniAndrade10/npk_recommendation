from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from datetime import datetime

from sql.insert_data import insert_data

import pickle
import os
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import pandas as pd


class model_creation:
    def __init__(
            self, df: pd.DataFrame, target_column: str, descr_columns: list
            ) -> None:
        self.df = df
        self.target_column = target_column
        self.descr_columns = descr_columns

        self.separate_data()

    def separate_data(self):
        self.X = self.df.loc[:, self.descr_columns]
        self.y = self.df.loc[:, self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2
        )

    def train_model(self, regressor: str, **kwargs):

        if regressor == "DecisionTreeRegressor":
            self.model = DecisionTreeRegressor(**kwargs)
            self.model.fit(self.X_train, self.y_train)
        elif regressor == "XGBRegressor":
            self.model = xgb.XGBRegressor(**kwargs)
            self.model.fit(self.X_train, self.y_train)
        else:
            print("It was not possible to create the model.")

    def check_train_performance(self):
        prediction = self.model.predict(self.X_train)
        self.result_r2 = r2_score(y_true=self.y_train, y_pred=prediction)

    def check_test_performance(self):
        prediction = self.model.predict(self.X_test)
        self.result_r2 = r2_score(y_true=self.y_test, y_pred=prediction)

    def get_previous_model(self, model_name):
        path_model = f"./src/model/{model_name}"
        if os.path.exists(path_model):
            with open(path_model, "rb") as model_file:
                self.prev_model = pickle.load(model_file)
        else:
            self.prev_model = None

    def check_test_performance_prev_model(self):
        if self.prev_model != None:
            prediction = self.prev_model.predict(self.X_test) # type:ignore
            self.prev_result_r2 = r2_score(y_true=self.y_test, y_pred=prediction)
        else:
            pass

    def write_information(self, model_name):
        self.time = str(datetime.now())
        data = [[self.time, model_name+self.time, self.result_r2]]
        cols = ["date", "model_name", "performance_r2"]
        df = pd.DataFrame(data=data, columns=cols)
        insert_data("./src/model/database/model_training.db", df)

    def write_model(self, model_name):
        self.write_information(model_name)

        path_model = f"./src/model/{model_name}"
        with open(path_model, "wb") as model_file:
            pickle.dump(self.model, model_file)

    def check_test_models(self, model_name):
        if self.prev_model == None:
            self.write_model(model_name)
        else:
            self.check_test_performance_prev_model()
            result_current = self.result_r2
            result_previous = self.prev_result_r2

            if result_current > result_previous:
                self.write_model(model_name)

    def run_allover(self, model_name):
        self.check_train_performance()
        self.check_test_performance()
        self.get_previous_model(model_name)
        self.check_test_performance_prev_model()
        self.check_test_models(model_name)


if __name__=="__main__":
    __ROOT_PATH__ = Path().resolve().parent
    file = "Crop_recommendation.csv"
    data_path = os.path.join(__ROOT_PATH__, "npk_recommendation", "data", file)
    df = pd.read_csv(data_path, sep=",")


    pdata_n = model_creation(df, "N", ["P", "temperature", "humidity", "ph", "rainfall"])
    pdata_p = model_creation(df, "P", ["temperature", "humidity", "ph", "rainfall"])
    pdata_k = model_creation(df, "K", ["N", "P", "temperature", "humidity", "ph", "rainfall"])

    model_n = pdata_n.train_model("XGBRegressor", learning_rate=0.1, max_depth=5, n_estimators=100)
    model_P = pdata_p.train_model("DecisionTreeRegressor", max_depth=15, min_samples_leaf=10)
    model_K = pdata_k.train_model("XGBRegressor", learning_rate=0.1, max_depth=5, n_estimators=200)

    pdata_n.run_allover("model_recommentation_n.pkl")
    pdata_p.run_allover("model_recommentation_p.pkl")
    pdata_k.run_allover("model_recommentation_k.pkl")
