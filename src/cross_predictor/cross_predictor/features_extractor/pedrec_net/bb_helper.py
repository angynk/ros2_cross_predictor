from typing import Union, Tuple, List
import numpy as np
import torch
from torch import Tensor
from data_structures import ImageSize



bb_type = Union[torch.tensor, np.ndarray, List[Union[float, int]]]

def get_bb_class_idx(bb: bb_type) -> int:
    """
    Returns the class index of the bounding box
    :param bb:
    :return:
    """
    return int(bb[5])


def get_bb_center(bb: bb_type) -> Tuple[float, float]:
    return bb[0], bb[1]


def get_bb_width(bb: bb_type) -> float:
    return bb[2]


def get_bb_height(bb: bb_type) -> float:
    return bb[3]


def bb_to_center_scale(bb: np.ndarray, input_size: ImageSize):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    bb : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    # center = np.zeros((2), dtype=np.float32)

    box_width = get_bb_width(bb)
    box_height = get_bb_height(bb)
    center = np.array(get_bb_center(bb))

    aspect_ratio = input_size.width / input_size.height

    if box_width > aspect_ratio * box_height:
        box_height = box_width / aspect_ratio
    else:
        box_width = box_height * aspect_ratio
    scale = np.array([box_width, box_height], dtype=np.float32) * 1.25

    return center, scale