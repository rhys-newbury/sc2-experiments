from pathlib import Path

import numpy as np
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from dash.exceptions import PreventUpdate
from konductor.metadata.database.sqlite import SQLiteDB, DEFAULT_FILENAME


class TimePoint:
    __slots__ = "value"

    def __init__(self, value: float):
        self.value = value

    def as_db_key(self):
        return "t_" + str(self.value).replace(".", "_")

    def as_pq_key(self):
        return "binary_acc_" + str(self.value)

    def as_float(self):
        return float(self.value)


time_points = [TimePoint(t) for t in np.arange(2, 20, 0.5)]


def get_labels():
    return {str(t.as_float()): "FLOAT" for t in time_points}


def pq_key_to_db_key(key: str):
    return "t_" + key.split("_")[-1].replace(".", "_")


layout = html.Div(
    children=[
        dbc.Row(
            [
                html.H3("Eval Accuracy Over % Game Length"),
                dcc.Dropdown(id="ts2-eval_folder", options=["tournament", "492"]),
                dcc.Graph(id="ts2-length-win", selectedData={}),
            ]
        ),
    ]
)


def hash_to_brief(root: Path):
    db_handle = SQLiteDB(root / DEFAULT_FILENAME)
    output = db_handle.cursor().execute("SELECT hash, brief FROM metadata").fetchall()
    return {x[0]: x[1] for x in output}


@callback(
    Output("ts2-length-win", "figure"),
    Input("ts2-eval_folder", "value"),
    Input("root-dir", "data"),
    prevent_initial_call=False,
)
def update_game_length(eval_folder: str, root: str):
    if not all([root]):
        raise PreventUpdate

    _root = Path(root)

    hb_map = hash_to_brief(_root)
    fig = go.Figure()

    for folder in _root.glob("*"):
        if not folder.is_dir():
            continue

        csv = folder / f"percent_{eval_folder}" / "game_length_results_50"

        if not csv.exists():
            continue

        with open(csv, encoding="utf-8") as f:
            lines = f.readlines()

        counts = lines[0].split(",")
        totals = lines[1].split(",")

        percentage = [float(c) / float(t) for c, t in zip(counts[1:], totals[1:])]
        times = list(range(0, 100, 2))

        # Add a scatter plot
        fig.add_trace(
            go.Scatter(
                x=times, y=percentage, mode="lines+markers", name=hb_map[folder.name]
            )
        )

    # Update layout
    fig.update_layout(
        title="Outcome forecast acc over Game Length",
        xaxis_title="% Game Length",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 1],
    )

    # Show the plot
    return fig
