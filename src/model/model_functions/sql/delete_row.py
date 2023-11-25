import pandas as pd
import os
from pathlib import Path
import sqlite3


def delete_row(path, id=None, col=None, value=None):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    if id != None:
        sql_delete_row = f"""
            DELETE from model_training WHERE id = {id}
                """
    elif col != None and value != None and value is not float:
        sql_delete_row = f"""
                    DELETE from model_training WHERE '{col}' = '{value}'
                        """
    elif col != None and value != None and value is float:
        sql_delete_row = f"""
                    DELETE from model_training WHERE '{col}' = {value}
                        """
    else:
        sql_delete_row = """ """

    cursor.execute(sql_delete_row)
    conn.commit()
    conn.close()


if __name__=="__main__":
    root_path = Path(__file__).resolve().parent.parent
    file = "model_training.db"
    path = os.path.join(root_path, "database", file)
    delete_row(path=path, id=4, col="model_name", value="Model_5")
