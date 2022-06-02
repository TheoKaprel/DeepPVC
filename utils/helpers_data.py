import itk
import torch
import numpy as np



# elif normalisation == "sum":
# data_sum = np.sum(dataset, axis=(2, 3))[:, :, None, None]
# dataset = dataset / data_sum
# elif normalisation == "mean":
# data_mean = np.mean(dataset, axis=(2, 3))[:, :, None, None]
# dataset = dataset / data_mean


def compute_norm(dataset, data_normalisation):
    if data_normalisation=="standard":
        mean = np.mean(dataset)
        std = np.std(dataset)
        norm = [mean, std]
    elif data_normalisation=="min_max":
        min = np.min(dataset)
        max = np.max(dataset)
        norm = [min, max]
    elif data_normalisation=="none":
        norm = None
    else:
        norm = None
    print(f'For norm type {data_normalisation}, the norm is {norm}')
    return norm

def normalize(dataset_or_img,normtype,norm, to_torch, device):
    if normtype=="standard":
        mean = norm[0]
        std = norm[1]
        out =  (dataset_or_img  - mean)/std
    elif normtype=="min_max":
        min = norm[0]
        max = norm[1]
        out =  (dataset_or_img - min)/(max - min)
    else:
        out = dataset_or_img
    if to_torch:
        out = torch.from_numpy(out).to(device)
    return out

def denormalize(dataset_or_img,normtype,norm, to_numpy):
    if normtype=="standard":
        mean = norm[0]
        std = norm[1]
        output = dataset_or_img*std + mean
    elif normtype=="min_max":
        min = norm[0]
        max = norm[1]
        output = dataset_or_img*(max - min) + min
    else:
        output = dataset_or_img
    if to_numpy:
        output = output.cpu().numpy()
    return output



def load_image(filename, is_ref):
    if is_ref:
        return load_PVE_PVfree(filename)
    else:
        return load_from_mhd(filename)


def load_from_mhd(filename_mhd):
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
    return img


def load_PVE_PVfree(ref):
    """
    1° Loads the two projections (PVE/PVfree)
    2° Concatenates both on first dim

    :param filename_mhd: input filename (.mhd)
    :return: If Input is a (N,N) projection, output is the normalised (1,1,N,N) tensor and the list of norms
    """

    proj_PVE_filename = f'{ref}_PVE.mhd'
    proj_PVfree_filename = f'{ref}_PVfree.mhd'

    imgPVE = load_from_mhd(proj_PVE_filename)

    imgPVfree = load_from_mhd(proj_PVfree_filename)
    array = np.concatenate((imgPVE, imgPVfree), axis=1)

    return array