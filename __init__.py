import numpy, torch
from typing import Tuple, TypedDict, Literal


class DictDataset(TypedDict, total=False):
    merge_x: numpy.ndarray | torch.Tensor
    merge_y: numpy.ndarray | torch.Tensor
    merge_x_mean: numpy.ndarray | torch.Tensor
    merge_y_mean: numpy.ndarray | torch.Tensor
    merge_x_std: numpy.ndarray | torch.Tensor
    merge_y_std: numpy.ndarray | torch.Tensor
    train_x: numpy.ndarray | torch.Tensor
    train_y: numpy.ndarray | torch.Tensor
    valid_x: numpy.ndarray | torch.Tensor
    valid_y: numpy.ndarray | torch.Tensor
