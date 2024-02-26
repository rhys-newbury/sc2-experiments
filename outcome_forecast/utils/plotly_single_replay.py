from pathlib import Path

import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from dash.exceptions import PreventUpdate

layout = html.Div(
    children=[
        html.H2("Replay Outcome Prediction"),
        dbc.Row(
            [
                dbc.Col(html.H4("Experiment Hash:")),
                dbc.Col(dcc.Dropdown(id="sr-experiments")),
                dbc.Col(html.H4("Replay:")),
                dbc.Col(dcc.Dropdown(id="sr-replay-hash")),
            ]
        ),
        dbc.Row([dcc.Graph(id="sr-replay-preds", selectedData={})]),
    ]
)


@callback(
    Output("sr-experiments", "options"),
    Input("root-dir", "data"),
    prevent_initial_call=False,
)
def update_runs(root: str):
    if not root:
        raise PreventUpdate()
    valid_runs = []
    for item in Path(root).iterdir():
        if (item / "outcome_prediction.csv").exists():
            valid_runs.append(item.stem)
    return valid_runs


@callback(
    Output("sr-replay-hash", "options"),
    Input("root-dir", "data"),
    Input("sr-experiments", "value"),
)
def update_replays(root: str, run_hash: str):
    """Update the list of replays to choose from"""
    if not root or not run_hash:
        raise PreventUpdate()

    data = pd.read_csv(Path(root) / run_hash / "outcome_prediction.csv")

    return list(data["replay"].unique())


@callback(
    Output("sr-replay-preds", "figure"),
    Input("root-dir", "data"),
    Input("sr-experiments", "value"),
    Input("sr-replay-hash", "value"),
)
def update_game_length(root: str, run_hash: str, replay_hash: str):
    if not all((root, run_hash, replay_hash)):
        raise PreventUpdate

    data = pd.read_csv(Path(root) / run_hash / "outcome_prediction.csv")
    col_to_rm = ["replay"]
    if "Unnamed: 0" in data.columns:
        col_to_rm.append("Unnamed: 0")
    data = data[data["replay"] == replay_hash].drop(columns=col_to_rm)

    aux_cols = {"playerId", "outcome"}
    times = pd.Series(
        [float(time) for time in filter(lambda x: x not in aux_cols, data.columns)],
        name="time [min]",
    )

    fig = go.Figure()
    winner = 9999
    for playerId in [1, 2]:
        # Add a scatter plot
        playerData = data[data["playerId"] == playerId]
        outcome = playerData.drop(columns=["playerId", "outcome"]).iloc[0]
        if playerData["outcome"].all():
            winner = playerId
        fig.add_trace(
            go.Scatter(
                x=times, y=outcome, mode="lines+markers", name=f"Player {playerId}"
            )
        )

    # Update layout
    fig.update_layout(
        title=f"Outcome Forecast, Player {winner} Win",
        xaxis_title="Game Time [min]",
        yaxis_title="Forecasted Win (%)",
        yaxis_range=[0, 1],
    )

    # Show the plot
    return fig
