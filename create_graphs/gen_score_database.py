import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Annotated

from sc2_replay_reader import Score
from summaryStats import SQL_TYPES, LambdaFunctionType, SC2Replay

app = typer.Typer()


def custom_collate(batch):
    # No read success in entire batch
    if not any(item["read_success"] for item in batch):
        return torch.utils.data.dataloader.default_collate(batch)

    if all(item["read_success"] for item in batch):
        return torch.utils.data.dataloader.default_collate(batch)

    first_read_success = next((item for item in batch if item.get("read_success")))
    extra_keys = set(first_read_success.keys()) - {"partition", "idx", "read_success"}

    # Create a dictionary with zero tensors for extra_keys
    empty_batch = {key: 0 for key in extra_keys}
    data_batch = [
        {**empty_batch, **data} if not data["read_success"] else data for data in batch
    ]

    return torch.utils.data.dataloader.default_collate(data_batch)


def make_database(
    path: Path,
    additional_columns: Dict[str, SQL_TYPES],
    features: Dict[str, SQL_TYPES],
    lambda_columns: Dict[str, Tuple[SQL_TYPES, LambdaFunctionType]],
):
    if path.exists():
        os.remove(path)
    # Connect to the SQLite database (creates a new database if it doesn't exist)
    conn = sqlite3.connect(str(path))

    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()

    # Create a table with the specified headings and data types
    create_table_sql = f"""
        CREATE TABLE game_data (
            {', '.join(f"{column} {datatype}" for column, datatype in additional_columns.items())},
            {', '.join(f"{column} {datatype}" for column, datatype in features.items())},
            {', '.join(f"{column} {datatype}" for column, (datatype, _) in lambda_columns.items())}
        )
    """
    cursor.execute(create_table_sql)
    return conn, cursor


def close_database(conn: sqlite3.Connection):
    # Commit the changes and close the connection
    conn.commit()
    conn.close()


def add_to_database(cursor: sqlite3.Cursor, data_dict: Dict[str, Any]):
    columns = ", ".join(data_dict.keys())
    placeholders = ", ".join("?" for _ in data_dict.values())

    query = f"""
        INSERT INTO game_data ({columns})
        VALUES ({placeholders})
    """

    cursor.execute(query, tuple(data_dict.values()))


@app.command()
def create_individual(
    workspace: Annotated[Path, typer.Option()] = Path("."),
    workers: Annotated[int, typer.Option()] = 0,
):
    # print(Path(os.environ["DATAPATH"]))
    for p in Path(os.environ["DATAPATH"]).glob("*.SC2Replays"):
        # print(p)
        try:
            main(workspace=workspace, workers=workers, name=p.name, replay=p)
        except Exception as e:
            print(e)
            print(f"failed, {p.name}")


@app.command()
def main(
    workspace: Annotated[Path, typer.Option()] = Path("."),
    workers: Annotated[int, typer.Option()] = 0,
    name: Annotated[str, typer.Option()] = "gamedata",
    replay: Annotated[Optional[Path], typer.Option()] = None,
):
    features: Dict[str, SQL_TYPES] = {
        "playerResult": "TEXT",
    }
    # Manually include additional columns
    additional_columns: Dict[str, SQL_TYPES] = {}

    all_attributes = [
        attr
        for attr in dir(Score)
        if not callable(getattr(Score, attr))
        if "__" not in attr
    ]

    lambda_columns: Dict[str, Tuple[SQL_TYPES, LambdaFunctionType]] = {
        "score": ("TEXT", lambda y: y.data.score),
        "gameStep": ("TEXT", lambda y: y.data.gameStep),
    }

    files = Path(os.environ["DATAPATH"]) if replay is None else replay
    dataset = SC2Replay(replay, set(features.keys()), lambda_columns)

    batch_size = 50
    scores_1 = []
    scores_0 = []
    for idx, d in tqdm(enumerate(dataset), total=len(dataset)):
        if d["playerResult"] == 1:
            scores_1.append(
                [
                    str(x.killed_value_units) + ";" + str(gs)
                    for x, gs in zip(d["score"], d["gameStep"])
                ]
            )
        if d["playerResult"] == 0:
            scores_0.append(
                [
                    str(x.killed_value_units) + ";" + str(gs)
                    for x, gs in zip(d["score"], d["gameStep"])
                ]
            )

        # print(d)
        # import pdb; pdb.set_trace()

        if idx % 100 == 0:
            with open("killed_value_units0.scores", "a") as f:
                f.write("\n".join([",".join(x) for x in scores_0]))
                f.write("\n")

            with open("killed_value_units1.scores", "a") as f:
                f.write("\n".join([",".join(x) for x in scores_1]))
                f.write("\n")
            scores_0 = []
            scores_1 = []

    with open("killed_value_units0.scores", "a") as f:
        f.write("\n".join([",".join(x) for x in scores_0]))
        f.write("\n")

    with open("killed_value_units1.scores", "a") as f:
        f.write("\n".join([",".join(x) for x in scores_1]))
        f.write("\n")


if __name__ == "__main__":
    app()
