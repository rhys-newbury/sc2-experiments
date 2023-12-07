from sklearn import preprocessing
import numpy as np
from src.data.replayFolder import SC2Replay, Split, TimeRange
import os
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import xgboost
import timeit

time = TimeRange(0, 6, 0.1)  # Minutes,

dataset = SC2Replay(
    Path(os.environ["DATAPATH"]),
    Split.TRAIN,
    0.8,
    {"minimap_features", "scalar_features"},
    time,
)
dataloader = DataLoader(dataset, batch_size=8, num_workers=8)

print("num batches: ", len(dataloader))
# Yeet the old files
for time_step, current_time in enumerate(time.arange()):
    file = f"./{time_step}_x.npy"
    wins = f"./{time_step}_y.npy"
    if Path(file).exists():
        os.remove(file)
        os.remove(wins)

t2 = timeit.default_timer()
# process and save as numpy files
for idx, sample in enumerate(dataloader):
    print(idx)
    print(f"Took {timeit.default_timer() - t2}s")
    t2 = timeit.default_timer()

    for time_step, current_time in enumerate(time.arange()):
        file = f"./{time_step}_x.npy"
        wins = f"./{time_step}_y.npy"

        if Path(file).exists():
            data = np.load(file)
            data = np.concatenate((data, sample["scalar_features"][:, time_step, :]))

            wins_array = np.load(wins)
            wins_array = np.concatenate((wins_array, sample["win"]))
            np.save(str(file), data)
            np.save(str(wins), wins_array)
        else:
            x = sample["scalar_features"][:, time_step, :]
            y = sample["win"]
            np.save(str(file), x)
            np.save(str(wins), y)

del dataloader
del dataset

for time_step, current_time in enumerate(time.arange()):
    print(f"Running current_time: {current_time}")
    file = f"./{time_step}_x.npy"
    wins = f"./{time_step}_y.npy"
    x = np.load(file)
    y = np.load(wins)
    X = preprocessing.StandardScaler().fit(x).transform(x)
    random_state = 22
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    print("Starting fit...")
    t1 = timeit.default_timer()
    xgb = xgboost.XGBClassifier(
        objective="binary:logistic",
        booster="gbtree",
        eta=0.2,
        max_depth=5,
        use_label_encoder=False,
        verbosity=0,
        random_state=random_state,
    )
    xgb_scores = cross_val_score(xgb, X, y, cv=cv, n_jobs=-1)
    t2 = timeit.default_timer()
    print(f"Took {t2 - t1}s")
    print(
        "XGBoost: %0.4f accuracy with a standard deviation of %0.4f"
        % (xgb_scores.mean(), xgb_scores.std())
    )
