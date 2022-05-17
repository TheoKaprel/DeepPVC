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


def load_tensor_PVE_PVfree_from_mhd(ref):
    """
    1° Loads the two projections (PVE/PVfree)
    2° Normalizes by max value
    3° Concatenates both on first dim
    4° Converts into tensor

    :param filename_mhd: input filename (.mhd)
    :return: If Input is a (N,N) projection, output is the normalised (1,1,N,N) tensor
    """

    proj_PVE_filename = f'{ref}_PVE.mhd'
    proj_PVfree_filename = f'{ref}_PVfree.mhd'

    imgPVE = itk.array_from_image(itk.imread(proj_PVE_filename))
    imgPVE = np.expand_dims(imgPVE, axis=0)
    max_img = np.max(imgPVE)
    imgPVE = imgPVE / max_img

    imgPVfree = itk.array_from_image(itk.imread(proj_PVfree_filename))
    imgPVfree = np.expand_dims(imgPVfree, axis=0)
    max_img = np.max(imgPVfree)
    imgPVfree = imgPVfree / max_img

    array = np.concatenate((imgPVE, imgPVfree), axis=1)

    tensor_img = torch.from_numpy(array)
    return tensor_img