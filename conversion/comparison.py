"""Compare cpu/mem usage between sc2-serializer and alphastar-unplugged"""

import os
import time
from pathlib import Path

import docker
import pandas as pd
import typer
from docker.models.containers import Container
from matplotlib import pyplot as plt

SC2_PATH = Path(os.environ.get("SC2_PATH", str(Path.home() / "SC2")))
REPLAYS_PATH = Path(os.environ.get("DATAPATH", "/mnt/dataset"))
ASTAR_FILENAME = "alphastar.csv"
SC2S_FILENAME = "sc2-serializer.csv"


def get_cpu_usage(stats) -> float:
    "returns CPU Usage as number of cores used"
    n_cpu = stats["cpu_stats"]["online_cpus"]

    def cpu_diff(s, key):
        return key(s["cpu_stats"]) - key(s["precpu_stats"])

    ctr_usage = cpu_diff(stats, lambda x: x["cpu_usage"]["total_usage"])
    sys_usage = cpu_diff(stats, lambda x: x["system_cpu_usage"])

    return ctr_usage / sys_usage * n_cpu


def get_memory_usage(stats) -> float:
    """returns memory usage in megabytes"""
    return stats["memory_stats"]["usage"] / 1e6


def run_stats(ctr: Container, logfile: Path):
    """Watch container stat usage and append to the logfile stem the time taken"""
    with open(logfile, "w", encoding="utf-8") as f:
        f.write("Time[sec],CPU[Core],MEM[MB]\n")

    stats = ctr.stats(stream=False)
    ctr.reload()
    while ctr.status != "exited":
        cpu = get_cpu_usage(stats)
        mem = get_memory_usage(stats)
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(f"{time.time()},{cpu},{mem}\n")
        stats = ctr.stats(stream=False)
        ctr.reload()

    print("Container finished")


app = typer.Typer()


@app.command()
def cpp_serializer(output: Path = Path.cwd()):
    """Run sc2-serializer in docker container and write results file to output directory"""
    client = docker.from_env()
    ctr = client.containers.run(
        "mu00120825.eng.monash.edu.au:5000/sc2-serializer",
        command=[
            "./sc2_converter",
            "--replays=/replays/4.9.2",
            "--partition=/data/parts/4.9.2/evaluation_partition",
            "--output=/data/converted/sc2_evaluation.SC2Replays",
            "--game=/data/game/4.9.2/Versions/",
            "--converter=action",
        ],
        volumes=[f"{SC2_PATH}:/data", f"{REPLAYS_PATH}:/replays"],
        detach=True,
    )
    run_stats(ctr, output / SC2S_FILENAME)


@app.command()
def alphastar(output: Path = Path.cwd()):
    """Run alphastar in docker container and write results file to output directory"""
    client = docker.from_env()
    ctr = client.containers.run(
        "mu00120825.eng.monash.edu.au:5000/smac-transformer:alphastar",
        command=[
            "python3",
            "data/generate_dataset.py",
            "--sc2_replay_path=/replays/4.9.2",
            "--converted_path=/data/converted/tfrec",
            "--partition_file=/data/parts/4.9.2/evaluation_partition",
            "--converter_settings=configs/alphastar_supervised_converter_settings.pbtxt",
            "--num_threads=1",
        ],
        working_dir="/home/worker/alphastar/unplugged",
        environment=["SC2PATH=/data/game/4.9.2"],
        volumes=[f"{SC2_PATH}:/data", f"{REPLAYS_PATH}:/replays"],
        detach=True,
    )
    run_stats(ctr, output / ASTAR_FILENAME)


@app.command()
def compare(results: Path = Path.cwd()):
    """Compare runs between sc2-serializer and alphastar"""
    cpp_stats = pd.read_csv(results / SC2S_FILENAME)
    astar_stats = pd.read_csv(results / ASTAR_FILENAME)
    x_key = "Time[sec]"
    for y_key in ["MEM[MB]", "CPU[Core]"]:
        plt.plot(
            cpp_stats[x_key] - cpp_stats[x_key][0],
            cpp_stats[y_key],
            label="sc2-serializer",
        )
        plt.plot(
            astar_stats[x_key] - astar_stats[x_key][0],
            astar_stats[y_key],
            label="alphastar",
        )
        plt.suptitle(
            f"sc2-serializer avg: {cpp_stats[y_key].mean():.2f}, max: {cpp_stats[y_key].max():.2f}"
            f"\nalphastar avg: {astar_stats[y_key].mean():.2f}, max: {astar_stats[y_key].max():.2f}"
        )
        plt.legend()
        plt.ylabel(y_key)
        plt.xlabel(x_key)
        plt.savefig(f"{y_key}.png")
        plt.close()


@app.command()
def update_time(results: Path = Path.cwd()):
    """Convert time to relative rather than absolute, writes modified files with '_t' suffix"""
    for filename in [SC2S_FILENAME, ASTAR_FILENAME]:
        filepath = results / filename
        stats = pd.read_csv(filepath)
        stats["Time[sec]"] -= stats["Time[sec]"][0]
        stats.to_csv(filepath.with_stem(filepath.stem + "_t"))


if __name__ == "__main__":
    app()
