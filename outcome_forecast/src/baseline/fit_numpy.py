import os
from enum import Enum
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch
import yaml
from konductor.utilities.pbar import IntervalPbar, LivePbar
from sklearn import preprocessing, svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader, Dataset
from typing_extensions import Annotated

try:
    import xgboost
except ImportError:
    xgboost = None

import typer

app = typer.Typer()


class Model(Enum):
    XGBOOST = "xgboost"
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"
    MLP = "mlp"


def get_model(m: Model, **config):
    def inner(random_state: int):
        match m:
            case Model.XGBOOST:
                if xgboost is None:
                    raise ImportError("Cannot Use XGBoost, as it is not installed :(")
                return xgboost.XGBClassifier(**config, random_state=random_state)
            case Model.MLP:
                return MLPClassifier(**config, random_state=random_state)
            case Model.SVM:
                return svm.SVC(**config, random_state=random_state)
            case Model.LOGISTIC_REGRESSION:
                return LogisticRegression(**config, random_state=random_state)

    return inner


class FolderDataset(Dataset):
    """
    Basic folder dataset (basepath/split/sample.npz) which contains numpy
    files that contain the expected format to train on directly
    """

    def __init__(self, basepath: Path) -> None:
        super().__init__()
        self.folder = basepath
        assert self.folder.exists(), f"Root folder doesn't exist: {basepath}"

        file_list = self.folder / "val-list.txt"
        with open(file_list, "r", encoding="utf-8") as f:
            self.files = ["val/" + s.strip() for s in f.readlines()]

        file_list = self.folder / "train-list.txt"
        with open(file_list, "r", encoding="utf-8") as f:
            self.files += ["train/" + s.strip() for s in f.readlines()]

        # self.files = self.files[:100]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        data = np.load(self.folder / self.files[index], allow_pickle=True)

        def transform(x: np.ndarray):
            return str(x) if "str" in x.dtype.name else torch.tensor(x)

        try:
            out_data = {k: transform(v) for k, v in data.items()}
        except BadZipFile as e:
            raise RuntimeError(f"Got bad data from {self.files[index]}") from e

        return out_data


@app.command()
def fit_model(
    model: Annotated[Model, typer.Option()] = Model.XGBOOST,
    yaml_config: Annotated[Path, typer.Option()] = Path.cwd() / "cfg" / "baseline.yml",
    num_workers: Annotated[int, typer.Option()] = 4,
    ts_index: Annotated[int, typer.Option()] = 0,
    batch_size: Annotated[int, typer.Option()] = 8,
    output: Annotated[Path, typer.Option()] = Path().cwd(),
    datapath: Annotated[Path, typer.Option()] = Path(
        "/run/user/1000/gvfs/smb-share:server=130.194.128.238,share=bryce-rhys/outcome-subset-492"
    ),
):
    if pn := os.environ.get("POD_NAME"):
        ts_index = int(pn.split("-")[-1])

    d = FolderDataset(datapath)
    liveProgress = False
    x = DataLoader(d, batch_size=batch_size, num_workers=num_workers, drop_last=False)

    if (output / f"{ts_index}.txt").exists():
        return

    assert yaml_config.exists()
    with yaml_config.open(mode="r") as file:
        yaml_data = yaml.safe_load(file)

    _m = get_model(model, **yaml_data[model.name])

    output_tensor = torch.zeros((len(d), 28))
    idx = 0
    gt_tensor = torch.zeros((len(d)))
    pbar_type = LivePbar if liveProgress else IntervalPbar
    pbar_kwargs = {"total": len(x), "desc": "recreate"}

    if not liveProgress:
        pbar_kwargs["fraction"] = 0.05

    with pbar_type(**pbar_kwargs) as pbar:
        for data in x:
            valid = data["valid"][:, ts_index]
            valid_data = data["scalar_features"][valid, ts_index, :]

            if valid.sum() > 1:
                output_tensor[idx : idx + valid.sum(), :] = valid_data
                gt_tensor[idx : idx + valid.sum()] = data["win"][valid]

            idx += valid.sum()
            pbar.update(batch_size)

        output_tensor = output_tensor[: idx + 1, :].numpy()
        gt_tensor = gt_tensor[: idx + 1].numpy()

        np.save(output / f"{ts_index}_data.npy", output_tensor)
        np.save(output / f"{ts_index}_gt.npy", gt_tensor)

        X = preprocessing.StandardScaler().fit(output_tensor).transform(output_tensor)
        random_state = 22
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        model_ = _m(random_state)
        model_scores = cross_val_score(model_, X, gt_tensor, cv=cv, n_jobs=-1)
        print(
            f"{model.name}: "
            f"{model_scores.mean():.4f} accuracy with a "
            f"standard deviation of {model_scores.std():.4f}"
        )
        with open(output / f"{ts_index}.txt", "w") as out_file:
            out_file.write(
                f"{model_scores.mean():.4f} accuracy with a standard deviation of "
                f"{model_scores.std():.4f}"
            )
