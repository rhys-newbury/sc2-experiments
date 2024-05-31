#!/usr/bin/env python3

import dash
from konductor.webserver.app import cliapp
from utils import plotly_comparison, plotly_gamelength, plotly_single_replay

dash.register_page(
    "Results Over Time",
    path="/timeseries-performance",
    layout=plotly_comparison.layout,
)

dash.register_page(
    "Results Over GameLength",
    path="/gamelength-performance",
    layout=plotly_gamelength.layout,
)

dash.register_page(
    "Replay Outcome Prediction",
    path="/replay-prediction",
    layout=plotly_single_replay.layout,
)

if __name__ == "__main__":
    cliapp()
