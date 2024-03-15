"""General testing of dataloading"""

import os
from pathlib import Path

from src.data.base_dataset import SC2ReplayOutcome, Split, TimeRange
from src.data.replay_sampler import SQLSampler


def test_iterating():
    sampler = SQLSampler(
        "$ENV:gamedata.db",
        Path(os.environ["DATAPATH"]),
        [
            "game_length > 6720",
            "read_success = 1",
            "parse_success = 1",
            "number_game_step > 1024",
            "playerAPM > 100",
        ],
        0.8,
        Split.TRAIN,
    )
    dataset = SC2ReplayOutcome(
        sampler,
        TimeRange(2, 30, 0.5),
        features={"minimap_features", "scalar_features"},
    )
    for idx, sample in enumerate(dataset):
        if idx == 100:
            break
        print(sample["win"])


if __name__ == "__main__":
    test_iterating()
