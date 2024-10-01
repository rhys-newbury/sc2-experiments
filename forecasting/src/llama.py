import os
from pathlib import Path

import typer
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from typing_extensions import Annotated
from sc2_serializer.sampler import SQLSampler
from sc2_serializer import set_replay_database_logger_level, spdlog_lvl
from data.torch_dataset import SC2ReplayOutcome, TimeRange

app = typer.Typer()

current_script_path = Path(__file__).resolve().parent


@app.command()
def run(
    yaml_config: Annotated[Path, typer.Option()] = current_script_path / "llama.yml",
    workers: Annotated[int, typer.Option()] = 8,
):
    assert yaml_config.exists()
    with yaml_config.open(mode="r") as file:
        yaml_data = yaml.safe_load(file)

    os.environ["DATAPATH"] = yaml_data["dataFolder"]

    time = TimeRange(**yaml_data["timepoints"])
    sampler = SQLSampler(
        yaml_data["database"],
        Path(os.environ["DATAPATH"]),
        yaml_data["sql_filters"],
        0.8,
        False,
    )

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)

    dataset = SC2ReplayOutcome(
        sampler, time, ["scalars"], metadata=True, load_other_player=True
    )

    # dataset.get_matching()
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=workers, collate_fn=collate_fn
    )

    for i in dataloader:
        if i is None:
            continue
        for ts in range(i["scalars"].shape[1]):
            for player in [0, 1]:
                [
                    score_float,
                    idle_production_time,
                    idle_worker_time,
                    total_value_units,
                    total_value_structures,
                    killed_value_units,
                    killed_value_structures,
                    collected_minerals,
                    collected_vespene,
                    collection_rate_minerals,
                    collection_rate_vespene,
                    spent_minerals,
                    spent_vespene,
                    total_damage_dealt_life,
                    total_damage_dealt_shields,
                    total_damage_dealt_energy,
                    total_damage_taken_life,
                    total_damage_taken_shields,
                    total_damage_taken_energy,
                    total_healed_life,
                    total_healed_shields,
                    total_healed_energy,
                    minearals,
                    vespene,
                    popMax,
                    popArmy,
                    popWorkers,
                    gameStep,
                ] = i["scalars"][0, ts, player, :]


if __name__ == "__main__":
    set_replay_database_logger_level(spdlog_lvl.warn)

    app()
