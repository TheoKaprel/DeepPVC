import itk
import torch
import numpy as np



def load_tensor_from_mhd(filename_mhd):
    """
    1° Loads the mhd projection
    2° Adds a dimension
    3° Normalizes by max value
    4° Converts into tensor

    :param filename_mhd: input filename (.mhd)
    :return: If Input is a (N,N) projection, output is the normalised (1,1,N,N) tensor
    """

    img = itk.array_from_image(itk.imread(filename_mhd))
    img = np.expand_dims(img, axis=0)
    max_img = np.max(img)
    img = img/max_img
    tensor_img = torch.from_numpy(img)
    return tensor_img