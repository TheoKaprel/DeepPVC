#!/usr/bin/env python3

import torch
from torch.nn.parallel import DistributedDataParallel


def init_data_parallelism(model):
    import idr_torch

    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")
    model.switch_device(gpu)

    if model.network_type == 'pix2pix':
        model.Generator = DistributedDataParallel(model.Generator, device_ids=[idr_torch.local_rank])
        model.Discriminator = DistributedDataParallel(model.Discriminator, device_ids=[idr_torch.local_rank])
    elif model.network_type == 'unet':
        model.UNet = DistributedDataParallel(model.UNet, device_ids=[idr_torch.local_rank])
    elif model.network_type=='unet_denoiser_pvc':
        model.UNet_denoiser = DistributedDataParallel(model.UNet_denoiser, device_ids=[idr_torch.local_rank])
        model.UNet_pvc = DistributedDataParallel(model.UNet_pvc, device_ids=[idr_torch.local_rank])
    elif model.network_type=='diffusion':
        model.Diffusion_Unet = DistributedDataParallel(model.Diffusion_Unet, device_ids=[idr_torch.local_rank])

def get_dataloader_params(dataset,jean_zay,split_dataset):
    if jean_zay:
        import idr_torch
        pin_memory = False
        number_gpu = idr_torch.size
        if split_dataset:
            sampler=None
            shuffle=True
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                        shuffle=True,
                                                                        num_replicas=idr_torch.size,
                                                                        rank=idr_torch.rank)
            shuffle = False

    else:
        number_gpu = torch.cuda.device_count()
        sampler,shuffle,pin_memory= None, True,False

    return sampler,shuffle,pin_memory,number_gpu


def get_gpu_id_nb_gpu(jean_zay):
    if jean_zay:
        import idr_torch
        return (idr_torch.rank,idr_torch.size)
    else:
        return (torch.cuda.current_device(), torch.cuda.device_count())