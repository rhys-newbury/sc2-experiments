#!/usr/bin/env python3
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from jinja2 import Environment, FileSystemLoader

app = typer.Typer()

VALID_GPUS = [
    "NVIDIA-GeForce-GTX-1080",
    "NVIDIA-GeForce-GTX-1080-Ti",
    "NVIDIA-GeForce-RTX-2080-Ti",
    "NVIDIA-GeForce-RTX-3090",
    "NVIDIA-RTX-A6000",
    "Tesla-T4",
]


@dataclass
class Dataset:
    subset: str
    path: str


DATASETS = [
    Dataset("tournament", "outcome-subset-tournament"),
    Dataset("492", "outcome-subset-492"),
    Dataset("tournament", "converted/tournaments"),
    Dataset("492", "converted/4.9.2"),
]


@app.command()
def main(
    exp_config: Annotated[str, typer.Option(help="experiment yaml to launch")],
    out_dir: Annotated[Path, typer.Option(help="output for rendered k8s")] = Path.cwd()
    / "manifest",
    template_name: Annotated[
        str, typer.Option(help="name of template to use")
    ] = "train.yml.j2",
):
    """Launch kubernetes job with path to training configuration"""

    template_conf: dict[str, str | int] = {
        "exp_config": exp_config,
        "registry": os.environ["REGISTRY_URL"],
    }
    template_conf["cpu"] = int(input("num cpu (4): ").strip() or "4") * 1000
    template_conf["mem"] = int(input("memory (Gi) (8): ").strip() or "8")
    template_conf["n_worker"] = int(input("num workers (4): ").strip() or "4")
    template_conf["n_gpu"] = int(input("n gpu (0): ").strip() or "0")
    if brief := input("brief (''): "):
        template_conf["brief"] = brief
    template_conf["epochs"] = int(input("epochs: "))
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], check=True, capture_output=True
    )
    template_conf["git_branch"] = proc.stdout.decode().strip()

    print("GPU Names Available")
    for idx, name in enumerate(VALID_GPUS):
        print(f"{idx}: {name}")
    try:
        template_conf["gpu_name"] = VALID_GPUS[
            int(input("Select GPU (empty to skip): "))
        ]
    except ValueError:
        pass

    print("Datasets (DATAPATH)")
    for idx, name in enumerate(DATASETS):
        print(f"{idx}: {name}")
    dataset = DATASETS[int(input("Select :"))]
    template_conf["dataset"] = dataset.path
    template_conf["dataset_type"] = dataset.subset

    env = Environment(loader=FileSystemLoader("template/"))
    template = env.get_template(template_name)

    kube_manifest = template.render(template_conf)
    out_path = (out_dir / exp_config).with_suffix(".yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(kube_manifest)

    subprocess.run(["kubectl", "apply", "-f", str(out_path)], check=True)


if __name__ == "__main__":
    app()
