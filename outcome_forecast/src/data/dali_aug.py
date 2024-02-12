"""Augmentations primarily for minimap data"""

from konductor.data.dali import DALI_AUGMENTATIONS
from nvidia.dali import fn
from nvidia.dali.types import DALIDataType
from nvidia.dali.data_node import DataNode


@DALI_AUGMENTATIONS.register_module("random-flip")
def random_flip(minimaps: DataNode, vertical: bool = True, horizontal: bool = True):
    """Random vertical and horizontal flipping"""
    if horizontal and fn.random.coin_flip(probability=0.5):
        minimaps = fn.flip(minimaps, horizontal=True)
    if vertical and fn.random.coin_flip(probability=0.5):
        minimaps = fn.flip(minimaps, vertical=True)
    return minimaps


@DALI_AUGMENTATIONS.register_module("random-rotate")
def random_rotate(minimaps: DataNode, angle: float):
    rand_angle = fn.random.uniform(range=[-angle, angle], dtype=DALIDataType.FLOAT)
    minimaps = fn.rotate(minimaps, angle=rand_angle)
