import itk
import torch
import numpy as np


def normalize(img, data_normalisation):
    norm=None
    if data_normalisation=='max':
        norm = np.max(img)
        img = img / norm
    elif data_normalisation=="sum":
        norm = np.sum(img)
        img = img / norm
    elif data_normalisation=="mean":
        norm = np.mean(img)
        img = img / norm
    elif data_normalisation=="min_max_glob":
        norm = 256
        img = img/norm
    elif data_normalisation=="none":
        norm = 1
        img = img/norm
    return img,norm

def load_image(filename,norm, is_ref):
    if is_ref:
        return load_tensor_PVE_PVfree_from_mhd(filename,norm)
    else:
        return load_tensor_from_mhd(filename,norm)


def load_tensor_from_mhd(filename_mhd, data_normalisation):
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
    img,norm = normalize(img,data_normalisation)
    tensor_img = torch.from_numpy(img)
    return tensor_img,norm





def load_tensor_PVE_PVfree_from_mhd(ref, data_normalisation):
    """
    1° Loads the two projections (PVE/PVfree)
    2° Normalizes by max value
    3° Concatenates both on first dim
    4° Converts into tensor

    :param filename_mhd: input filename (.mhd)
    :return: If Input is a (N,N) projection, output is the normalised (1,1,N,N) tensor and the list of norms
    """

    proj_PVE_filename = f'{ref}_PVE.mhd'
    proj_PVfree_filename = f'{ref}_PVfree.mhd'

    imgPVE = itk.array_from_image(itk.imread(proj_PVE_filename))
    imgPVE = np.expand_dims(imgPVE, axis=0)
    imgPVE,normPVE = normalize(imgPVE,data_normalisation)


    imgPVfree = itk.array_from_image(itk.imread(proj_PVfree_filename))
    imgPVfree = np.expand_dims(imgPVfree, axis=0)
    imgPVfree,normPVfree = normalize(imgPVfree,data_normalisation)

    array = np.concatenate((imgPVE, imgPVfree), axis=1)

    tensor_img = torch.from_numpy(array)
    return tensor_img,[normPVE, normPVfree]