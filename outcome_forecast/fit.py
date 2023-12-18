from sklearn import preprocessing
import numpy as np
from src.data import SC2Replay, Split, TimeRange
import os
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

import xgboost
import timeit
import torch
import matplotlib.pyplot as plt
import typer
from typing_extensions import Annotated
from enum import Enum
import yaml

app = typer.Typer()

current_script_path = Path(__file__).resolve().parent


class Model(Enum):
    XGBOOST = "xgboost"
    MLP = "mlp"


def get_model(m: Model, **config):
    def inner(random_state: int):
        match m:
            case Model.XGBOOST:
                return xgboost.XGBClassifier(
                    **config,
                    random_state=random_state,
                )
            case Model.MLP:
                return MLPClassifier(**config, random_state=random_state)

    return inner


@app.command()
def fit_model(
    model: Annotated[Model, typer.Option()] = "xgboost",
    yaml_config: Annotated[Path, typer.Option()] = current_script_path
    / "cfg"
    / "baseline.yml",
    recreate_split: Annotated[bool, typer.Option()] = False,
    min_game_time: Annotated[int, typer.Option()] = 5,
    tmp_workspace: Annotated[Path, typer.Option()] = Path("./processed_data"),
    save_plots: Annotated[bool, typer.Option()] = True,
):
    time = TimeRange(0, 15, 0.3)  # Minutes,

    dataset = SC2Replay(
        Path(os.environ["DATAPATH"]),
        Split.TRAIN,
        0.8,
        {"minimap_features", "scalar_features"},
        time,
        min_game_time=min_game_time,
    )
    dataloader = DataLoader(dataset, batch_size=8, num_workers=8)

    if recreate_split:
        for time_step, current_time in enumerate(time.arange()):
            file = tmp_workspace / f"{time_step}_x.npy"
            wins = tmp_workspace / f"{time_step}_y.npy"
            if file.exists():
                os.remove(file)
                os.remove(wins)

        t2 = timeit.default_timer()
        # process and save as numpy files
        for idx, sample in enumerate(dataloader):
            print(idx)
            print(f"Took {timeit.default_timer() - t2}s")
            t2 = timeit.default_timer()

            difference_array = np.absolute(time.arange() - min_game_time)

            # find the index of minimum element from the array
            five_minute_index = difference_array.argmin()
            mask = sample["valid"].sum(axis=1) > five_minute_index

            for time_step, current_time in enumerate(time.arange()):
                file = tmp_workspace / f"{time_step}_x.npy"
                wins = tmp_workspace / f"{time_step}_y.npy"

                mask_ = torch.logical_and(sample["valid"][:, time_step], mask)
                x = sample["scalar_features"][:, time_step, :][mask_, :]
                y = sample["win"][mask_]

                if mask_.sum() == 0:
                    break

                if file.exists():
                    data = np.load(file)
                    data = np.concatenate((data, x))

                    wins_array = np.load(wins)
                    wins_array = np.concatenate((wins_array, y))
                    np.save(str(file), data)
                    np.save(str(wins), wins_array)
                else:
                    np.save(str(file), x)
                    np.save(str(wins), y)

        del dataloader
        del dataset

    results = []
    std_dev = []
    assert yaml_config.exists()
    with yaml_config.open(mode="r") as file:
        yaml_data = yaml.safe_load(file)

    _m = get_model(model, **yaml_data[model.name])

    for time_step, current_time in enumerate(time.arange()):
        print(f"Running current_time: {current_time}")
        file = tmp_workspace / f"{time_step}_x.npy"
        wins = tmp_workspace / f"{time_step}_y.npy"

        if not (file.exists() and wins.exists()):
            break

        x = np.load(file)
        y = np.load(wins)
        print(f"for timestep = {time_step}, shape of data is {x.shape}")
        X = preprocessing.StandardScaler().fit(x).transform(x)
        random_state = 22
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        print("Starting fit...")
        t1 = timeit.default_timer()

        model_ = _m(random_state)

        model_scores = cross_val_score(model_, X, y, cv=cv, n_jobs=-1)
        t2 = timeit.default_timer()
        results.append(model_scores.mean())
        std_dev.append(model_scores.std())
        print(f"Took {t2 - t1}s")
        print(
            f"{model.name}: "
            f"{model_scores.mean():.4f} accuracy with a "
            f"standard deviation of {model_scores.std():.4f}"
        )

    if save_plots:
        plt.plot((time.arange() * 60).tolist(), results, label="Results")

        plt.fill_between(
            (time.arange() * 60).tolist(),
            [result - std for result, std in zip(results, std_dev)],
            [result + std for result, std in zip(results, std_dev)],
            alpha=0.2,
            label="Standard Deviation",
        )

        # Add labels and title
        plt.xlabel("Time (s)")
        plt.ylabel("Accuracry")
        plt.title(f"Accuracy of {model.name}")

        # Add legend
        plt.legend()
        plt.savefig(f"{model.name}.png")

    return results, std_dev


@app.command()
def fit_all(
    yaml_config: Annotated[Path, typer.Option()] = current_script_path
    / "cfg"
    / "baseline.yml",
    recreate_split: Annotated[bool, typer.Option()] = False,
    min_game_time: Annotated[int, typer.Option()] = 5,
    tmp_workspace: Annotated[Path, typer.Option()] = Path("./processed_npy_files"),
    save_plots: Annotated[bool, typer.Option()] = True,
):
    results = {}
    for m in Model:
        results[m.name] = fit_model(
            model=m,
            yaml_config=yaml_config,
            recreate_split=recreate_split,
            min_game_time=min_game_time,
            tmp_workspace=tmp_workspace,
            save_plots=False,
        )

        recreate_split = False

    if save_plots:
        time = TimeRange(0, 15, 0.3)  # Minutes,

        for name, (results, std_dev) in results.items():
            line_color = np.random.rand(
                3,
            )

            plt.plot(
                (time.arange() * 60).tolist(), results, label=name, color=line_color
            )

            plt.fill_between(
                (time.arange() * 60).tolist(),
                [result - std for result, std in zip(results, std_dev)],
                [result + std for result, std in zip(results, std_dev)],
                alpha=0.2,
                color=line_color,
                label=f"{name}_Standard Deviation",
            )

        # Add labels and title
        plt.xlabel("Time (s)")
        plt.ylabel("Accuracry")

        # Add legend
        plt.legend()

        plt.savefig("all.png")


if __name__ == "__main__":
    app()
