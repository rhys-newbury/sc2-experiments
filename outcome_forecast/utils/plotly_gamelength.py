from pathlib import Path

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from dash.exceptions import PreventUpdate
from konductor.metadata.database.sqlite import SQLiteDB, DEFAULT_FILENAME

layout = html.Div(
    children=[
        dbc.Row(
            [
                html.H3("Eval Accuracy Over % Game Length"),
                dcc.Dropdown(id="ts2-eval-folder", options=["tournament", "492"]),
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
    Input("ts2-eval-folder", "value"),
    Input("root-dir", "data"),
    prevent_initial_call=False,
)
def update_game_length(eval_folder: str, root_: str):
    if not root_:
        raise PreventUpdate()

    root = Path(root_)

    hb_map = hash_to_brief(root)
    fig = go.Figure()

    for folder in filter(lambda x: x.is_dir(), root.iterdir()):
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
        title="Outcome forecast acc over Game Duration",
        xaxis_title="% Game Duration",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 1],
    )

    # Show the plot
    return fig
