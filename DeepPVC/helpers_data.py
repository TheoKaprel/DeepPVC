import itk
import torch
import numpy as np


def compute_norm(dataset, data_normalisation):
    if data_normalisation=="standard":
        mean = np.mean(dataset)
        std = np.std(dataset)
        norm = [mean, std]
    elif data_normalisation in ["min_max", "min_max_1_1"] :
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
    elif normtype=="min_max_1_1":
        min = norm[0]
        max = norm[1]
        out = (2 * (dataset_or_img - min) )/(max - min) -1
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
    elif normtype == "min_max_1_1":
        min = norm[0]
        max = norm[1]
        output = ((dataset_or_img+1)*( max - min ) + 2 * min ) / 2
    else:
        output = dataset_or_img
    if to_numpy:
        output = output.cpu().numpy()
    return output



def load_image(filename, is_ref, type, noisy=False):
    if is_ref:
        return load_PVE_PVfree(ref = filename, type=type, noisy=noisy)
    else:
        return load_from_filename(filename)


def load_from_filename(filename):
    img = itk.array_from_image(itk.imread(filename))
    img = np.expand_dims(img, axis=(1,2)) # (nb_proj,1,1,128,128)
    return img


def load_PVE_PVfree(ref, type, noisy):

    proj_PVE_filename = f'{ref}_PVE.{type}'
    proj_PVfree_filename = f'{ref}_PVfree.{type}'

    imgPVE = load_from_filename(proj_PVE_filename)

    imgPVfree = load_from_filename(proj_PVfree_filename)

    if noisy:
        proj_PVE_noisy_filename = f'{ref}_PVE_noisy.{type}'
        imgPVE_noisy = load_from_filename(proj_PVE_noisy_filename)
        array = np.concatenate((imgPVE_noisy, imgPVE, imgPVfree), axis=1) # (1,3,nb_channels,128,128)
    else:
        array = np.concatenate((imgPVE, imgPVfree), axis=1) # (1,2,nb_channels,128,128)

    return array