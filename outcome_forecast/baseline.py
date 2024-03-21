#!/usr/bin/env python3
import typer

from src.baseline import fit, fit_numpy, minimap

app = typer.Typer()
app.add_typer(fit.app, name="fit")
app.add_typer(fit_numpy.app, name="fit-numpy")
app.add_typer(minimap.app, name="minimap")

if __name__ == "__main__":
    app()
