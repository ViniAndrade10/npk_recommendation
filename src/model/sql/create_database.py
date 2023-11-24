import pandas as pd
import os
from pathlib import Path
import sqlite3


def creating_db(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    sql_create = """
            CREATE TABLE IF NOT EXISTS model_training
            (
            id INTEGER PRIMARY KEY,
            date DATETIME NOT NULL,
            model_name TEXT NOT NULL,
            performance_r2 FLOAT NOT NULL
            )
                """

    cursor.execute(sql_create)
    conn.commit()
    conn.close()


if __name__=="__main__":
    breakpoint()
    root_path = Path(__file__).resolve().parent.parent
    file = "model_training.db"
    path = os.path.join(root_path, "database", file)
    creating_db(path=path)
