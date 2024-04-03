import itk
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader, TensorDataset

from . import dataset as pvc_dataset

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
        out = dataset_or_img - np.mean(dataset_or_img[:,0:1,:,:,:], axis = (1,2,3), keepdims=True)
        out = out / np.std(dataset_or_img[:,0:1,:,:,:], keepdims=True)
    elif normtype=="img_0_1":
        min_per_img = np.min(dataset_or_img[:,0:1,:,:,:], axis=(2,3,4), keepdims=True)
        max_per_img = np.max(dataset_or_img[:,0:1,:,:,:], axis=(2,3,4), keepdims=True)
        out = (dataset_or_img - min_per_img) / (max_per_img - min_per_img)
    elif normtype=="img_mean":
        if dataset_or_img.ndim==5:
            out = dataset_or_img / np.mean(dataset_or_img[:,0,:,:,:], axis = (1,2,3))[:,None,None,None,None]
        else:
            out = dataset_or_img / np.mean(dataset_or_img[0,:,:,:])
    else:
        out = dataset_or_img

    if to_torch:
        out = torch.tensor(out, device=device)
    return out


def compute_norm_eval(dataset_or_img, data_normalisation):
    if ('global' in data_normalisation or data_normalisation=='none'):
        norm = None
    elif data_normalisation == 'img_standard':
        mean = torch.mean(dataset_or_img[:, 0:1, :, :, :], dim=(1, 2, 3,4), keepdim=True)
        std = torch.std(dataset_or_img[:, 0:1, :, :, :], dim=(1, 2, 3,4), keepdim=True)
        norm = [mean, std]
    elif data_normalisation in ['img_0_1', 'img_1_1']:
        min = torch.amin(dataset_or_img[:, 0:1, :, :, :],dim=(1, 2, 3,4),keepdim=True)
        max = torch.amax(dataset_or_img[:, 0:1, :, :, :],dim=(1, 2, 3,4),keepdim=True)
        norm = [min, max]
    elif data_normalisation == 'img_mean':
        mean = torch.mean(dataset_or_img, dim=(1, 2, 3))
        norm = [mean]
    elif data_normalisation=="3d_max":
        if "rec_fp" in dataset_or_img.keys():
            max = torch.amax(dataset_or_img['rec_fp'], dim=(1,2,3), keepdim=False)
        elif "rec" in dataset_or_img.keys():
            max = torch.amax(dataset_or_img['rec'], dim=(1,2,3), keepdim=False)
        else:
            max = torch.amax(dataset_or_img['PVE_noisy'], dim=(1,2,3), keepdim=False)
        return [max]
    elif data_normalisation=="3d_mean":
        mean = torch.mean(dataset_or_img[0], dim=(1,2,3), keepdim=False)
        return [mean]
    elif data_normalisation=="3d_std":
        mean = torch.mean(dataset_or_img[0], dim=(1,2,3), keepdim=False)
        std = torch.std(dataset_or_img[0], dim=(1,2,3), keepdim=False)
        return [mean, std]
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
        mean_per_img = norm[0][:,None,None,None]
        out = dataset_or_img / mean_per_img
    elif data_normalisation=="3d_max":
        max= norm[0]
        for key in dataset_or_img.keys():
            if key=="attmap_fp":
                max_attmap = torch.amax(dataset_or_img['attmap_fp'], dim=(1, 2, 3), keepdim=False)
                max_attmap[max_attmap==0]=1 # avoids nan after division by max
                dataset_or_img[key] = dataset_or_img[key]/ max_attmap[:,None,None,None]
            elif key=="lesion_mask":
                pass
            else:
                dataset_or_img[key] = dataset_or_img[key] / max[:,None,None,None]
        out = dataset_or_img
    elif data_normalisation=="3d_mean":
        mean= norm[0]
        if type(dataset_or_img)==tuple:
            out = tuple([input_i/mean[:,None,None,None] for input_i in dataset_or_img])
        else:
            out = dataset_or_img / mean[:,None,None,None]
    elif data_normalisation=="3d_std":
        mean,std= norm[0], norm[1]
        if type(dataset_or_img)==tuple:
            out = tuple([(input_i - mean[:,None,None,None]) / std[:,None,None,None] for input_i in dataset_or_img])
        else:
            out = (dataset_or_img - mean[:,None,None,None]) / std[:,None,None,None]
    else:
        out = dataset_or_img

    if to_torch:
        device = torch.device(params['device'])
        out = torch.tensor(out, device=device)
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
        mean = norm[0][:,None,None,None]
        output = mean*dataset_or_img
    elif data_normalisation=="3d_max":
        max = norm[0]
        output = dataset_or_img * max[:,None,None,None]
    elif data_normalisation=="3d_mean":
        mean = norm[0]
        output = dataset_or_img * mean[:,None,None,None]
    elif data_normalisation=="3d_std":
        mean,std= norm[0], norm[1]
        output = (dataset_or_img * std[:,None,None,None] + mean[:,None,None,None])
    else:
        output = dataset_or_img

    return output


def load_image(filename, is_ref, type,params):
    if is_ref:
        return load_PVE_PVfree(ref = filename,type=type,params=params)
    else:
        return load_from_filename(filename, params)


def load_from_filename(filename,params):
    if filename[-3:]=="npy":
        img=np.load(filename)[None,:,:,:]
    else:
        img = itk.array_from_image(itk.imread(filename))[None,:,:,:]
    nb_projs=img.shape[1]
    input_eq_angles=params['input_eq_angles']
    with_adj_angles=params['with_adj_angles']

    nb_channels = input_eq_angles + 2 if with_adj_angles else input_eq_angles

    # channels_id construction
    nb_of_equidistributed_angles = input_eq_angles
    step = int(nb_projs / (nb_of_equidistributed_angles))
    channels_id = np.array([0])
    if with_adj_angles:
        adjacent_channels_id = np.array([(-1) % nb_projs, (1) % nb_projs])
        channels_id = np.concatenate((channels_id, adjacent_channels_id))
    equiditributed_channels_id = np.array(
        [(k * step) % nb_projs for k in range(1, nb_of_equidistributed_angles)])
    channels_id = np.concatenate((channels_id, equiditributed_channels_id)) if len(
        equiditributed_channels_id) > 0 else channels_id


    input_img = np.zeros((nb_projs, nb_channels,img.shape[2], img.shape[3]))
    for proj_i in range(nb_projs):
        channels_id_proj_i = (channels_id+proj_i)%120
        input_img[proj_i,:,:,:] = img[:,channels_id_proj_i,:,:]
    return torch.Tensor(input_img)


def load_PVE_PVfree(ref, type,params):
    noisy=params['with_noise']

    proj_PVE_filename = f'{ref}_PVE.{type}'
    proj_PVfree_filename = f'{ref}_PVfree.{type}'

    imgPVE = load_from_filename(proj_PVE_filename,params)
    imgPVfree = load_from_filename(proj_PVfree_filename,params)

    if noisy:
        proj_PVE_noisy_filename = f'{ref}_PVE_noisy.{type}'
        imgPVE_noisy = load_from_filename(proj_PVE_noisy_filename,params) # (120,6,256,256)
        with_rec_fp=params['with_rec_fp']
        if with_rec_fp:
            proj_rec_fp_filename = f'{ref}_rec_fp.{type}'
            img_rec_fp = torch.Tensor(itk.array_from_image(itk.imread(proj_rec_fp_filename))[:,None,:,:]) # (120,1,256,256)
            img_input = torch.cat((imgPVE_noisy,img_rec_fp),dim=1) # (120,7,256,256)
            return TensorDataset(img_input, imgPVfree[:, 0:1, :, :])
        else:
            return TensorDataset(imgPVE_noisy, imgPVfree[:,0:1,:,:])
    else:
        return TensorDataset(imgPVE, imgPVfree[:,0:1,:,:])


def build_channels_id(sino,nb_projs_per_img,input_eq_angles=None,nb_sino=None,with_adj_angles=None):
    # rotating channels id
    channels_id = np.array([0])
    if sino:
        adjacent_channels_id = np.array([(-k) % nb_projs_per_img for k in range(1, nb_sino // 2 + 1)] +
                                        [(k) % nb_projs_per_img for k in range(1, nb_sino // 2 + 1)])
        channels_id = np.concatenate((channels_id, adjacent_channels_id))
    else:
        step = int(nb_projs_per_img / (input_eq_angles))
        # adj angles
        if with_adj_angles:
            adjacent_channels_id = np.array([(-1) % nb_projs_per_img, (1) % nb_projs_per_img])
            channels_id = np.concatenate((channels_id, adjacent_channels_id))
        # eq angles
        equiditributed_channels_id = np.array(
            [(k * step) % nb_projs_per_img for k in range(1, input_eq_angles)])
        channels_id = np.concatenate((channels_id, equiditributed_channels_id)) if len(
            equiditributed_channels_id) > 0 else channels_id
    return channels_id


def get_channels_id_i(channels_id,proj_i,nb_projs_per_img):
    return (channels_id + proj_i) % nb_projs_per_img

def build_channels_id_(sino,nb_projs_per_img,input_eq_angles=None,nb_sino=None,with_adj_angles=None,nb_adj_angles=None):
    # rotating channels id
    channels_id = np.array([0])
    if sino:
        adjacent_channels_id = np.array([(-k) % nb_projs_per_img for k in range(1, nb_sino // 2 + 1)] +
                                        [(k) % nb_projs_per_img for k in range(1, nb_sino // 2 + 1)])
        channels_id = np.concatenate((channels_id, adjacent_channels_id))
    else:
        step = int(nb_projs_per_img / (input_eq_angles))

        # adj angles
        if with_adj_angles:
            # adjacent_channels_id = np.array([(-1) % self.nb_projs_per_img, (1) % self.nb_projs_per_img])

            adjacent_channels_id = np.array([
                k%nb_projs_per_img for k in range(-nb_adj_angles, nb_adj_angles)
            ])

            channels_id = adjacent_channels_id

        # eq angles
        equiditributed_channels_id = np.array([(k * step) % nb_projs_per_img for k in range(1, input_eq_angles)])
        channels_id = np.concatenate((channels_id, equiditributed_channels_id)) if len(
            equiditributed_channels_id) > 0 else channels_id
    return channels_id


def get_dataset_for_eval(params,input_PVE_noisy_array, input_rec_fp_array=None, attmap_fp_array=None):
    with_rec_fp = params['with_rec_fp']
    if "with_att" in params:
        with_att = params['with_att']
    else:
        with_att = False

    if params['inputs']=="projs":
        if params['sino']:
            sino=True
            nb_sino = params['input_eq_angles']
            nb_projs_per_img, nb_pix_x, nb_pix_y = input_PVE_noisy_array.shape[1], input_PVE_noisy_array.shape[0],input_PVE_noisy_array.shape[2]
            with_adj_angles = None
            input_eq_angles = None
        else:
            sino = False
            with_adj_angles = params["with_adj_angles"]
            input_eq_angles = params['input_eq_angles']
            nb_adj_angles = params['nb_adj_angles']

            nb_sino=None
            nb_projs_per_img, nb_pix_x, nb_pix_y = input_PVE_noisy_array.shape[0], input_PVE_noisy_array.shape[1], input_PVE_noisy_array.shape[2]
        pad = torch.nn.ConstantPad2d((0, 0, 4, 4), 0)

        channels_id = build_channels_id_(sino=sino, nb_projs_per_img=nb_projs_per_img,
                                        input_eq_angles=input_eq_angles, nb_sino=nb_sino,
                                        with_adj_angles=with_adj_angles,
                                        nb_adj_angles=nb_adj_angles)

        print(channels_id)

        data_PVE_noisy, data_rec_fp, data_attmap_fp = None,None,None
        for proj_i in range(nb_projs_per_img):

            channels_i = get_channels_id_i(channels_id=channels_id, proj_i=proj_i, nb_projs_per_img=nb_projs_per_img)


            data_PVE_noisy_i = torch.Tensor(input_PVE_noisy_array[channels_i,:,:])

            if with_rec_fp:
                data_rec_fp_i = torch.Tensor(input_rec_fp_array)[channels_i,:,:]
            if with_att:
                data_attmap_fp_i = torch.Tensor(attmap_fp_array)[channels_i,:,:]


            if data_PVE_noisy is None:
                data_PVE_noisy = data_PVE_noisy_i[None,:,:,:]
                if with_rec_fp:
                    data_rec_fp = data_rec_fp_i[None,:,:,:]
                if with_att:
                    data_attmap_fp = data_attmap_fp_i[None,:,:,:]
            else:
                data_PVE_noisy = torch.concatenate((data_PVE_noisy,data_PVE_noisy_i[None,:,:,:]),dim=0)
                if with_rec_fp:
                    data_rec_fp = torch.concatenate((data_rec_fp,data_rec_fp_i[None,:,:,:]),dim=0)
                if with_att:
                    data_attmap_fp = torch.concatenate((data_attmap_fp,data_attmap_fp_i[None,:,:,:]),dim=0)


        data_inputs = {}
        data_inputs['PVE_noisy'] = data_PVE_noisy
        if with_rec_fp:
            data_inputs['rec_fp'] = data_rec_fp
        if with_att:
            data_inputs['attmap_fp'] = data_attmap_fp
        return data_inputs

    elif params['inputs']=='full_sino':
        sino = params['sino']
        nb_projs_per_img, nb_pix_x, nb_pix_y = input_PVE_noisy_array.shape[0], input_PVE_noisy_array.shape[1],input_PVE_noisy_array.shape[2]
        data_PVE_noisy = torch.from_numpy(input_PVE_noisy_array.astype(np.float32))

        if params['pad']=="zero":
            pad = torch.nn.ConstantPad2d((0, 0, 4, 4), 0) if sino else torch.nn.ConstantPad2d((0, 0, 0, 0, 4, 4), 0)
        elif params['pad']=="circular":
            pad = pvc_dataset.CircularPadSino(4)
        else:
            pad = torch.nn.Identity()


        data_inputs = {}
        data_inputs['PVE_noisy'] = data_PVE_noisy
        if with_rec_fp:
            data_inputs['rec_fp'] = torch.from_numpy(input_rec_fp_array.astype(np.float32))
        if with_att:
            data_inputs['attmap_fp'] = torch.from_numpy(attmap_fp_array.astype(np.float32))

        for key_inputs in data_inputs.keys():
            data_inputs[key_inputs] = pad(data_inputs[key_inputs])[None,:,:,:]


        return data_inputs

    elif params['inputs']=="imgs":
        if params['pad']=="zero":
            pad = pvc_dataset.ZeroPadImgs(112)
        else:
            pad = torch.nn.Identity()

        data_PVE_noisy = pad(torch.from_numpy(input_PVE_noisy_array[None,:,:,:].astype(np.float32)))
        data_inputs = {}
        data_inputs['rec'] = data_PVE_noisy
        if with_att:
            data_inputs['attmap_rec_fp'] = pad(torch.from_numpy(attmap_fp_array[None,:,:,:].astype(np.float32)))

        return data_inputs


def back_to_input_format(params,output):
    if params['inputs']=="projs":
        output_array = output
        if params['sino']:
            output_array = output_array.transpose((1,0,2))[4:124,:,:]
    elif params['inputs']=="full_sino":
        output_array = output.cpu().numpy().squeeze()
        if params['sino']:
            output_array = output_array.transpose((1,0,2))[4:124,:,:]
    elif params['inputs']=="imgs":
        output_array=output.cpu().numpy().squeeze()
    return output_array



def get_data_for_eval_(params,input_PVE_noisy_array, input_rec_fp_array):
    with_rec_fp = params['with_rec_fp']

    if ('sino' in params and not params['full_sino']):
        # sino
        # projs_input_ = itk.array_from_image(itk.imread(input_fn))  # (120,256,256)
        nb_sino = params['sino']
        projs_input_t = projs_input_.transpose((1, 0, 2))  # (256,120,256)
        nb_projs_per_img = projs_input_t.shape[0]
        adjacent_channels_id = np.array([0] + [(-k) % nb_projs_per_img for k in range(1, nb_sino // 2 + 1)] +
                                        [(k) % nb_projs_per_img for k in range(1, nb_sino // 2 + 1)])

        projs_input = np.zeros((nb_projs_per_img, nb_sino + 1, projs_input_t.shape[-2], projs_input_t.shape[-1]))
        for proj_i in range(nb_projs_per_img):
            proj_channels = (adjacent_channels_id + proj_i) % nb_projs_per_img
            projs_input[proj_i] = projs_input_t[proj_channels, :, :]
        # (256, 7,120,256)
        if with_rec_fp:
            projs_rec_fp = itk.array_from_image(itk.imread(input_rec_fp_fn)).transpose((1, 0, 2))[:, None, :,
                           :]  # (256,1,120,256)
            projs_input = np.concatenate((projs_input, projs_rec_fp), axis=1)

        zeros_padding = np.zeros((projs_input.shape[0], projs_input.shape[1], 4, projs_input.shape[3]))
        projs_input = np.concatenate((zeros_padding, projs_input, zeros_padding), axis=2)

        projs_input = torch.Tensor(projs_input)
        # end sino

    elif params['full_sino']:
        projs_input = torch.Tensor(itk.array_from_image(itk.imread(input)))[None, :, :, :]
        if 'sino' in params:
            pad = torch.nn.ConstantPad2d((0, 0, 4, 4), 0)
            projs_input = pad(projs_input.transpose((0, 2, 1, 3)))

        if with_rec_fp:
            img_rec_fp = torch.Tensor(itk.array_from_image(itk.imread(input_rec_fp))[None, :, :, :])
            if 'sino' in params:
                img_rec_fp = pad(img_rec_fp.transpose((0, 2, 1, 3)))

            data_input = (projs_input, img_rec_fp)
        else:
            data_input = (projs_input)
    else:
        data_input = load_image(filename=input, is_ref=False, type=None, params=params)
        if with_rec_fp:
            img_rec_fp = torch.Tensor(
                itk.array_from_image(itk.imread(input_rec_fp))[:, None, :, :])  # (120,1,256,256)
            data_input = torch.cat((data_input, img_rec_fp), dim=1)  # (120,7,256,256)


