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
import json
import tqdm


app = typer.Typer()
current_script_path = Path(__file__).resolve().parent

# Helper function to convert game steps to a human-readable time (e.g., 5:32)
def game_step_to_time(game_step):
    total_seconds = round(float(game_step) / 22.4)  # Convert game steps to seconds
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"  # Format as mm:ss


@app.command()
def run(
    yaml_config: Annotated[Path, typer.Option()] = current_script_path / "llama.yml",
    workers: Annotated[int, typer.Option()] = 8,
    ckpt_dir: str = "/mnt/pretrained/LLaMA-2/llama-2-7b-chat/",
    tokenizer_path: str = "/mnt/pretrained/LLaMA-2/tokenizer.model",
    output_dir: str = "/mnt/fast/"  # Path to your writable SMB share (change to fast if desired)
):
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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

    dataloader = DataLoader(
        dataset, batch_size=2, num_workers=workers, collate_fn=collate_fn
    )

    max_game_steps = 15 * 60 * 22.4  # 15 minutes * 60 seconds * 22.4 game steps/second

    game_number = 0
    for i in tqdm.tqdm(dataloader,total=100):
        game_number += 1
        print(f"Processing game number: {game_number}")

        if i is None:
            continue

        game_data = []
        for ts in range(i["scalars"].shape[2]):
            game_step = i["scalars"][0, 0, ts, -1]
            # if game_step > max_game_steps:
            #     print(f"Reached game step {game_step}, stopping collection at 15 minutes.")
            #     break

            for player in [0, 1]:
                player_data = {
                    "game_step": game_step_to_time(game_step),
                    "player": player,
                    "scalars": {
                        "score_float": float(i["scalars"][0, player, ts, 0]),
                        "idle_production_timgame_step": float(i["scalars"][0, player, ts, 1]),
                        "idle_worker_time": float(i["scalars"][0, player, ts, 2]),
                        "total_value_units": float(i["scalars"][0, player, ts, 3]),
                        "total_value_structures": float(i["scalars"][0, player, ts, 4]),
                        "killed_value_units": float(i["scalars"][0, player, ts, 5]),
                        "killed_value_structures": float(i["scalars"][0, player, ts, 6]),
                        "collected_minerals": float(i["scalars"][0, player, ts, 7]),
                        "collected_vespene": float(i["scalars"][0, player, ts, 8]),
                        "collection_rate_minerals": float(i["scalars"][0, player, ts, 9]),
                        "collection_rate_vespene": float(i["scalars"][0, player, ts, 10]),
                        "spent_minerals": float(i["scalars"][0, player, ts, 11]),
                        "spent_vespene": float(i["scalars"][0, player, ts, 12]),
                        "total_damage_dealt_life": float(i["scalars"][0, player, ts, 13]),
                        "total_damage_dealt_shields": float(i["scalars"][0, player, ts, 14]),
                        "total_damage_dealt_energy": float(i["scalars"][0, player, ts, 15]),
                        "total_damage_taken_life": float(i["scalars"][0, player, ts, 16]),
                        "total_damage_taken_shields": float(i["scalars"][0, player, ts, 17]),
                        "total_damage_taken_energy": float(i["scalars"][0, player, ts, 18]),
                        "total_healed_life": float(i["scalars"][0, player, ts, 19]),
                        "total_healed_shields": float(i["scalars"][0, player, ts, 20]),
                        "total_healed_energy": float(i["scalars"][0, player, ts, 21]),
                        "minerals": float(i["scalars"][0, player, ts, 22]),
                        "vespene": float(i["scalars"][0, player, ts, 23]),
                        "popMax": float(i["scalars"][0, player, ts, 24]),
                        "popArmy": float(i["scalars"][0, player, ts, 25]),
                        "popWorkers": float(i["scalars"][0, player, ts, 26]),
                        "gameStep": float(i["scalars"][0, player, ts, 27])
                    }
                }
                game_data.append(player_data)

        # Write game data to JSON in your SMB share
        output_file = output_path / f"game_data_{game_number}.json"
        full_game = {"win": i["win"].cpu().numpy().tolist(), "game":game_data}
        with open(output_file, "w") as outfile:
            json.dump(full_game, outfile, indent=4)  # Using indent=4 for pretty printing
        print(f"Game data saved to {output_file}")
        if game_number == 100:
            break

if __name__ == "__main__":
    set_replay_database_logger_level(spdlog_lvl.warn)
    app()
