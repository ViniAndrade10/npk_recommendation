import pandas as pd
import os
from pathlib import Path
import sqlite3


def delete_data(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    sql_delete = """
            DELETE FROM model_training
                """
    cursor.execute(sql_delete)

    conn.commit()
    conn.close()


if __name__=="__main__":
    root_path = Path(__file__).resolve().parent.parent
    file = "model_training.db"
    path = os.path.join(root_path, "database", file)
    delete_data(path=path)
