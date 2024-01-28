from pathlib import Path
from typing import Annotated

import typer
from jinja2 import Environment, FileSystemLoader
from kubernetes import client, config, utils

app = typer.Typer()

VALID_GPUS = [
    "NVIDIA-GeForce-GTX-1080",
    "NVIDIA-GeForce-GTX-1080-Ti",
    "NVIDIA-GeForce-RTX-2080-Ti",
    "NVIDIA-GeForce-RTX-3090",
    "NVIDIA-RTX-A6000",
    "Quadro-GV100",
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

    template_conf: dict[str, str | int | None] = {"exp_config": exp_config}
    template_conf["cpu"] = int(input("num cpu (8): ").strip() or "8") * 1000
    template_conf["mem"] = int(input("memory (Gi) (16): ").strip() or "16")
    template_conf["n_worker"] = int(input("num workers (8): ").strip() or "8")
    template_conf["n_gpu"] = int(input("n gpu (1): ").strip() or "1")
    if brief := input("brief (''): "):
        template_conf["brief"] = brief
    template_conf["epochs"] = int(input("epochs: "))

    print("GPU Names Available")
    for idx, name in enumerate(VALID_GPUS):
        print(f"{idx}: {name}")
    template_conf["gpu_name"] = VALID_GPUS[int(input("Select GPU: "))]

    env = Environment(loader=FileSystemLoader("template/"))
    template = env.get_template(template_name)

    kube_manifest = template.render(template_conf)
    out_path = out_dir / exp_config
    with open(out_path.with_suffix(".yaml"), "w", encoding="utf-8") as f:
        f.write(kube_manifest)

    config.load_kube_config()

    k8s_client = client.ApiClient()
    utils.create_from_yaml(k8s_client, out_path)


if __name__ == "__main__":
    app()
