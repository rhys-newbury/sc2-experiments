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
x = DataLoader(dataset, batch_size=len(dataset))

for idx, sample in enumerate(x):
    print("Running xgboost with batches: ", sample["scalar_features"].shape[0])
    for time_step, current_time in enumerate(time.arange()):
        print(f"Running current_time: {current_time}")
        x = sample["scalar_features"][:, time_step, :]
        y = sample["win"]
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
