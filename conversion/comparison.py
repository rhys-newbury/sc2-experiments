from pathlib import Path
import time

import docker
import typer
from docker.models.containers import Container


def get_cpu_usage(stats) -> float:
    "returns CPU Usage as number of cores used"
    n_cpu = stats["cpu_stats"]["online_cpus"]

    def cpu_diff(s, key):
        return key(s["cpu_stats"]) - key(s["precpu_stats"])

    ctrUsage = cpu_diff(stats, lambda x: x["cpu_usage"]["total_usage"])
    sysUsage = cpu_diff(stats, lambda x: x["system_cpu_usage"])

    return ctrUsage / sysUsage * n_cpu


def get_memory_usage(stats) -> float:
    """returns memory usage in megabytes"""
    return stats["memory_stats"]["usage"] / 1e6


def run_stats(ctr: Container, logfile: Path):
    """Watch container stat usage and append to the logfile stem the time taken"""
    stime = time.time()
    with open(logfile, "w", encoding="utf-8") as f:
        f.write("CPU[Core],MEM[MB]\n")

    while ctr.status == "created":
        stats = ctr.stats(stream=False)
        cpu = get_cpu_usage(stats)
        mem = get_memory_usage(stats)
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(f"{cpu},{mem}\n")
    duration = time.time() - stime

    # Suffix the file with the time taken for conversion
    new_filename = logfile.with_stem(f"{logfile.stem}_{int(duration)}sec")
    logfile.rename(new_filename)


app = typer.Typer()
SC2_PATH = Path.home() / "SC2"


@app.command()
def cpp_serializer(results: Path = Path.cwd()):
    client = docker.from_env()
    ctr = client.containers.run(
        "mu00120825.eng.monash.edu.au:5000/sc2-serializer",
        command=[
            "./sc2_converter",
            "--replays=/data/replays/4.9.2",
            "--partition=/data/parts/4.9.2/evaluation_partition",
            "--output=/data/converted/sc2_evaluation.SC2Replays",
            "--game=/data/game/4.9.2/Versions/",
            "--converter=action",
        ],
        volumes=[f"{SC2_PATH}:/data"],
        detach=True,
    )
    run_stats(ctr, results / "sc2-serializer.txt")


@app.command()
def alphastar(results: Path = Path.cwd()):
    client = docker.from_env()
    ctr = client.containers.run(
        "mu00120825.eng.monash.edu.au:5000/smac-transformer:alphastar",
        command=[
            "python3",
            "data/generate_dataset.py",
            "--sc2_replay_path=/data/replays/4.9.2",
            "--converted_path=/data/converted/4.9.2",
            "--partition_file=/data/parts/4.9.2/evaluation_partition",
            "--converter_settings=configs/alphastar_supervised_converter_settings.pbtxt",
            "--num_threads=1",
        ],
        working_dir="/home/worker/alphastar/unplugged",
        environment=["SC2PATH=/data/game/4.9.2"],
        volumes=[f"{SC2_PATH}:/data"],
        detach=True,
    )
    run_stats(ctr, results / "alphastar.txt")


if __name__ == "__main__":
    app()
