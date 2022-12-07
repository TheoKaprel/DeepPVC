import itk
import torch
import numpy as np


def compute_norm(dataset, data_normalisation):
    if ("global" not in data_normalisation):
        norm=None
    else:
        if data_normalisation=="global_standard":
            mean = np.mean(dataset[:,0,:,:,:])
            std = np.std(dataset[:,0,:,:,:])
            norm = [mean,std]
        elif data_normalisation=="global_0_1":
            min = np.min(dataset[:,0,:,:,:])
            max = np.max(dataset[:,0,:,:,:])
            norm = [min,max]
        else:
            print(f"ERROR in data_normalisation : {data_normalisation}")
            exit(0)

    print(f'For norm type {data_normalisation}, the norm is {norm}')
    return norm

def normalize(dataset_or_img,normtype,norm, to_torch, device):
    if normtype=="global_standard":
        mean = norm[0]
        std = norm[1]
        out =  (dataset_or_img  - mean)/std
    elif normtype=="global_0_1":
        min = norm[0]
        max = norm[1]
        out =  (dataset_or_img - min)/(max - min)
    elif normtype=="img_standard":
        out = dataset_or_img - np.mean(dataset_or_img[:,0,:,:,:][:,None,:,:,:], axis = (1,2,3), keepdims=True)
        out = out / np.std(dataset_or_img[:,0,:,:,:][:,None,:,:,:], keepdims=True)
    elif normtype=="img_0_1":
        min_per_img = np.min(dataset_or_img[:,0,:,:,:][:,None,:,:,:], axis=(2,3,4), keepdims=True)
        max_per_img = np.max(dataset_or_img[:,0,:,:,:][:,None,:,:,:], axis=(2,3,4), keepdims=True)
        out = (dataset_or_img - min_per_img) / (max_per_img - min_per_img)
    elif normtype=="img_mean":
        mean_per_img = np.mean(dataset_or_img[:,0,:,:,:][:,None,:,:,:], axis = (2,3,4), keepdims=True)
        out = dataset_or_img / mean_per_img
    else:
        out = dataset_or_img

    if to_torch:
        out = torch.from_numpy(out).to(device)
    return out


def compute_norm_eval(dataset_or_img, data_normalisation):
    if ('global' in data_normalisation or data_normalisation=='none'):
        norm = None
    elif data_normalisation == 'img_standard':
        mean = np.mean(dataset_or_img[:, 0, :, :, :][:,None,:,:,:], axis=(1, 2, 3,4), keepdims=True)
        std = np.std(dataset_or_img[:, 0, :, :, :][:,None,:,:,:], axis=(1, 2, 3,4), keepdims=True)
        norm = [mean, std]
    elif data_normalisation == 'img_0_1':
        min = np.min(dataset_or_img[:, 0, :, :, :][:,None,:,:,:], axis=(1, 2, 3,4), keepdims=True)
        max = np.max(dataset_or_img[:, 0, :, :, :][:,None,:,:,:], axis=(1, 2, 3,4), keepdims=True)
        norm = [min, max]
    elif data_normalisation == 'img_mean':
        mean = np.mean(dataset_or_img[:, 0, :, :, :][:,None,:,:,:], axis=(1, 2, 3,4), keepdims=True)
        norm = [mean]
    else:
        print(f"ERROR in data_normalisation : {data_normalisation}")
        exit(0)
    return norm


def normalize_eval(dataset_or_img, data_normalisation, norm, params, to_torch):
    if data_normalisation=="global_standard":
        norm = params['norm']
        mean = norm[0]
        std = norm[1]
        out =  (dataset_or_img  - mean)/std
    elif data_normalisation=="global_0_1":
        norm = params['norm']
        min = norm[0]
        max = norm[1]
        out =  (dataset_or_img - min)/(max - min)
    elif data_normalisation=="img_standard":
        mean = norm[0]
        std = norm[1]
        out = dataset_or_img - mean
        out = out / std
    elif data_normalisation=="img_0_1":
        min = norm[0]
        max = norm[1]
        out =  (dataset_or_img - min)/(max - min)
    elif data_normalisation=="img_mean":
        mean_per_img = norm[0]
        out = dataset_or_img / mean_per_img
    else:
        out = dataset_or_img

    if to_torch:
        device = torch.device(params['device'])
        out = torch.from_numpy(out).to(device)
    return out

def denormalize_eval(dataset_or_img, data_normalisation, norm, params, to_numpy):
    if to_numpy:
        dataset_or_img = dataset_or_img.cpu().numpy()

    if data_normalisation=="global_standard":
        norm = params['norm']
        mean = norm[0]
        std = norm[1]
        output = dataset_or_img*std + mean
    elif data_normalisation=="global_0_1":
        norm = params['norm']
        min = norm[0]
        max = norm[1]
        output = min + (max-min)*dataset_or_img
    elif data_normalisation=="img_standard":
        mean = norm[0][:,0,:,:,:]
        std = norm[1][:,0,:,:,:]
        output = dataset_or_img*std + mean
    elif data_normalisation=="img_0_1":
        min = norm[0][:,0,:,:,:]
        max = norm[1][:,0,:,:,:]
        output = min + (max-min)*dataset_or_img
    elif data_normalisation=="img_mean":
        mean = norm[0][:,0,:,:,:]
        output = mean*dataset_or_img
    else:
        output = dataset_or_img

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