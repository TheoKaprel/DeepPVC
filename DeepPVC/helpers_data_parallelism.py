#!/usr/bin/env python3

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def init_data_parallelism(model):
    import idr_torch

    dist.init_process_group(backend='nccl', init_method='env://', world_size=idr_torch.size, rank=idr_torch.rank)
    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")
    model.switch_device(gpu)
    model.Generator = DistributedDataParallel(model.Generator, device_ids=[idr_torch.local_rank])
    model.Discriminator = DistributedDataParallel(model.Discriminator, device_ids=[idr_torch.local_rank])


def get_dataloader_params(dataset,batch_size,jean_zay,split_dataset):
    if jean_zay:
        import idr_torch
        batch_size_per_gpu = batch_size // idr_torch.size
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
        sampler,shuffle,batch_size_per_gpu,pin_memory= None, True,batch_size,False

    return sampler,shuffle,batch_size_per_gpu,pin_memory,number_gpu


def get_gpu_id_nb_gpu(jean_zay):
    if jean_zay:
        import idr_torch
        return (idr_torch.rank,idr_torch.size)
    else:
        return (torch.cuda.current_device(), torch.cuda.device_count())