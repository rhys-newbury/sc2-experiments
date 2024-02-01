#!/usr/bin/env python3

import dash
from konductor.webserver.app import cliapp
from utils import plotly_comparison

dash.register_page(
    "Results over time", path="/timeseries-performance", layout=plotly_comparison.layout
)

if __name__ == "__main__":
    cliapp()
