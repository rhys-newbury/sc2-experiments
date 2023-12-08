import sqlite3
from sklearn import preprocessing
import numpy as np
from src.data.summaryStats import SC2Replay, LambdaFunctionType
import os
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import xgboost
import timeit
import torch
from sklearn.model_selection import GridSearchCV
from sc2_replay_reader import (
    ReplayParser, Score)
from typing import Dict, Callable, Tuple
import typer
from typing_extensions import Annotated
app = typer.Typer()

def make_database(path, additional_columns, features, lambda_columns):
    # Connect to the SQLite database (creates a new database if it doesn't exist)
    conn = sqlite3.connect(str(path))

    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()

    # Create a table with the specified headings and data types
    create_table_sql = f'''
        CREATE TABLE game_data (
            {', '.join(f"{column} {datatype}" for column, datatype in additional_columns.items())},
            {', '.join(f"{column} {datatype}" for column, datatype in features.items())},
            {', '.join(f"{column} {datatype}" for column, (datatype, _) in lambda_columns.items())}

        )
    '''
    print(create_table_sql)
    cursor.execute(create_table_sql)
    return conn, cursor

def close_database(conn):
    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def add_to_database(cursor, data_dict):
    columns = ', '.join(data_dict.keys())
    placeholders = ', '.join('?' for _ in data_dict.values())

    query = f'''
        INSERT INTO game_data ({columns})
        VALUES ({placeholders})
    '''
    print(query)

    cursor.execute(query, tuple(data_dict.values()))

@app.command()
def main(workspace : Annotated[Path, typer.Option()], workers: Annotated[int, typer.Option()]):
    
    features = {
        "replayHash": "TEXT", "gameVersion": "TEXT", "playerId": "INTEGER", "playerRace": "TEXT", "playerResult": "TEXT", "playerMMR": "INTEGER", "playerAPM": "INTEGER"
    }
    # Manually include additional columns
    additional_columns = {
        "partition": "TEXT",
        "idx": "INTEGER"
    }
    all_attributes = [attr for attr in dir(Score) if not callable(getattr(Score, attr)) if "__" not in attr]

    lambda_columns : Dict[str, Tuple[str, LambdaFunctionType]] = {
        "max_units": ("TEXT", lambda y : max(len(x) for x in  y.data.units)),
        **{f"final_{i}": ("FLOAT", (lambda k: lambda y, key=k: float(getattr(y.data.score[-1], key)))(i)) for i in all_attributes}
    }

    conn, cursor = make_database(workspace / "gamedata.db", additional_columns, features, lambda_columns)

    dataset = SC2Replay(
        Path(os.environ["DATAPATH"]),
        set(features.keys()),
        lambda_columns
    )
    dataloader = DataLoader(dataset, num_workers=workers, batch_size=1)
    for idx, d in enumerate(dataloader):
        converted_d = {}

        for key, value in d.items():
            if isinstance(value, torch.Tensor):
                # If the value is a tensor, convert it to its numeric value
                converted_d[key] = value[0].item()
            elif isinstance(value, str):
                # If the value is a string, keep it as is
                converted_d[key] = value[0]
        if idx % 5 == 0:
            conn.commit()

        add_to_database(cursor, converted_d)
        
if __name__ == "__main__":
    app()