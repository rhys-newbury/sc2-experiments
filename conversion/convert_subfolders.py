"""Runs over a bunch of folder"""

import concurrent.futures as fut
import subprocess
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

app = typer.Typer()


def make_args(game: Path, replays: Path, converter: Path, output: Path, port: int):
    return [
        str(converter.absolute()),
        f"--replays={replays}",
        f"--output={output}",
        "--converter=action",
        f"--game={game}",
        f"--port={port}",
    ]


@app.command()
def main(
    converter: Annotated[Path, typer.Option()],
    outfolder: Annotated[Path, typer.Option()],
    replays: Annotated[Path, typer.Option()],
    game: Annotated[Path, typer.Option()],
    n_parallel: Annotated[int, typer.Option()] = cpu_count() // 2,
    extra_args: Annotated[Optional[list[str]], typer.Option()] = None,
):
    """"""
    assert converter.exists()
    assert game.exists()
    assert replays.exists()
    assert (
        0 < n_parallel <= 2 * cpu_count()
    ), f"{n_parallel=}, does not satisfy 0 < n_parallel < 2 * cpu_count()"
    outfolder.mkdir(exist_ok=True)

    with fut.ProcessPoolExecutor(max_workers=n_parallel) as ctx:
        res = []
        for idx, folder in enumerate(replays.iterdir()):
            if folder.is_file():
                continue
            res.append(
                ctx.submit(
                    subprocess.run,
                    make_args(
                        game,
                        folder,
                        converter,
                        outfolder / (folder.stem + ".SC2Replays"),
                        9168 + idx,
                    ),
                )
            )

        fut.wait(res)


if __name__ == "__main__":
    app()
