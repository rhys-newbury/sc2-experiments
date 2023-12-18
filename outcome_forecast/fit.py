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

app = typer.Typer()


class model(Enum):
    XGBOOST = "xgboost"
    MLP = "mlp"


def get_model(m: model):
    def inner(random_state: int):
        match m:
            case model.XGBOOST:
                return xgboost.XGBClassifier(
                    objective="binary:logistic",
                    booster="gbtree",
                    eta=0.2,
                    max_depth=5,
                    use_label_encoder=False,
                    verbosity=0,
                    random_state=random_state,
                )
            case model.MLP:
                return MLPClassifier(
                    hidden_layer_sizes=(
                        100,
                    ),  # You can customize the hidden layer structure
                    activation="relu",
                    solver="adam",  # Solver for weight optimization
                    alpha=0.0001,  # L2 regularization term
                    learning_rate_init=0.001,  # Initial learning rate
                    max_iter=1000,  # Maximum number of iterations
                    random_state=random_state,  # Random seed for reproducibility
                )

    return inner


@app.command()
def fit_model(
    model: Annotated[model, typer.Option()],
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
        min_game_time = 5
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
    _m = get_model(model)

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

        xgb = _m(random_state)

        xgb_scores = cross_val_score(xgb, X, y, cv=cv, n_jobs=-1)
        t2 = timeit.default_timer()
        results.append(xgb_scores.mean())
        std_dev.append(xgb_scores.std())
        print(f"Took {t2 - t1}s")
        print(
            "XGBoost: %0.4f accuracy with a standard deviation of %0.4f"
            % (xgb_scores.mean(), xgb_scores.std())
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
        plt.xlabel("X Axis Label")
        plt.ylabel("Y Axis Label")
        plt.title("Line Plot with Standard Deviation")

        # Add legend
        plt.legend()
        plt.savefig("yeet.png")


if __name__ == "__main__":
    app()
