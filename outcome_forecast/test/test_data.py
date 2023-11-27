"""General testing of dataloading"""
import os
from pathlib import Path

from src.data.replayFolder import SC2Replay, Split


def test_iterating():
    dataset = SC2Replay(
        Path(os.environ["DATAPATH"]),
        Split.TRAIN,
        0.8,
        {"minimap_features", "scalar_features"},
        2,
        30,
    )
    for idx, sample in enumerate(dataset):
        if idx == 100:
            break
        print(sample["win"])


if __name__ == "__main__":
    test_iterating()
