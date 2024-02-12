#!/usr/bin/env python3

import dash
from konductor.webserver.app import cliapp
from utils import plotly_comparison
from utils import plotly_gamelength

dash.register_page(
    "Results over time", path="/timeseries-performance", layout=plotly_comparison.layout
)

dash.register_page(
    "Results over GameLength",
    path="/gamelength-performance",
    layout=plotly_gamelength.layout,
)

if __name__ == "__main__":
    cliapp()
