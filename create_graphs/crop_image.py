#!/usr/bin/env python3
from pathlib import Path

import typer
from PIL import Image

app = typer.Typer()


@app.command()
def main(path: Path, top: int, left: int, bottom: int, right: int):
    """Save crop of image, good idea to save tlbr params for reuse"""
    im = Image.open(path).crop((left, top, right, bottom))
    im.save(Path.cwd() / f"{path.stem}_crop.png")


if __name__ == "__main__":
    app()
