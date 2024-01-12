import numpy as np
from plyfile import PlyData, PlyElement
import torch
import cv2

def save_ply(array: np.ndarray, filename: str):
    assert array.ndim == 2
    assert array.shape[1] == 3
    PlyData([PlyElement.describe(np.array([tuple(p) for p in array], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex')], text=True).write(filename)

def save_img(array: np.ndarray, filename: str):
    if type(array) == torch.Tensor:
        array = array.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[-1] == 3:
        img = cv2.cvtColor((array * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    elif array.ndim == 3 and array.shape[0] == 3:
        return save_img(array.transpose(1, 2, 0), filename)
    elif array.ndim == 2:
        img = (array * 255).astype(np.uint8)
    elif array.ndim == 3 and array.shape[0] == 1:
        return save_img(array[0], filename)
    else:
        print(array.shape)
        raise NotImplementedError
    return cv2.imwrite(filename, img)
