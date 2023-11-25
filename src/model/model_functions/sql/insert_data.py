import pandas as pd
import os
from pathlib import Path
import sqlite3


def insert_data(path, df: pd.DataFrame):
    conn = sqlite3.connect(path)
    df.to_sql("model_training", con=conn, index=False, if_exists="append")
    conn.close()


if __name__=="__main__":
    root_path = Path(__file__).resolve().parent.parent
    file = "model_training.db"
    path = os.path.join(root_path, "database", file)

    df = pd.DataFrame([["20-11-2023", "Model_5", 0.885]]).rename(columns={0:"date", 1:"model_name", 2:"performance_r2"})
    insert_data(path=path, df=df)
