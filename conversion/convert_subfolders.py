"""Runs over a bunch of folder"""

import concurrent.futures as fut
import subprocess
import io
from multiprocessing import cpu_count
from pathlib import Path

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


def run_with_redirect(tid, *args):
    """Redirect each thread stdout to log file"""
    print(f"running {tid}: {args}")
    with open(f"worker_logs_{tid}.txt", "a", encoding="utf-8") as f:
        subprocess.run(args, stdout=io.TextIOWrapper(f.buffer, write_through=True))


@app.command()
def main(
    converter: Annotated[Path, typer.Option()],
    outfolder: Annotated[Path, typer.Option()],
    replays: Annotated[Path, typer.Option()],
    game: Annotated[Path, typer.Option()],
    n_parallel: Annotated[int, typer.Option()] = cpu_count() // 2,
):
    """"""
    assert converter.exists()
    assert game.exists()
    assert replays.exists()
    assert (
        0 < n_parallel <= cpu_count()
    ), f"{n_parallel=}, does not satisfy 0 < n_parallel < cpu_count()"
    outfolder.mkdir(exist_ok=True)

    with fut.ProcessPoolExecutor(max_workers=n_parallel) as ctx:
        res = []
        for idx, folder in enumerate(replays.iterdir()):
            if folder.is_file():
                continue
            outfile = outfolder / (folder.stem + ".SC2Replays")
            args = make_args(game, folder, converter, outfile, 9168 + idx)
            res.append(ctx.submit(run_with_redirect, idx, *args))
        fut.wait(res)


if __name__ == "__main__":
    app()
