import numpy as np
import torch

from . import helpers_data
from torch.cuda.amp import autocast
import torch.distributed as dist

def validation_errors(test_dataloader, model, do_NRMSE=True, do_NMAE=True):
    MNRMSE,std_NRMSE = 0,0
    MNMAE,std_NMAE = 0,0

    data_normalisation = model.params['data_normalisation']
    device = model.device

    list_NRMSE = torch.Tensor([0.]).to(device)
    list_NMAE = torch.Tensor([0.]).to(device)

    with torch.no_grad():
        for test_it,(batch_inputs,batch_targets) in enumerate(test_dataloader):
            with autocast():
                batch_inputs = batch_inputs.to(device,non_blocking=True)
                batch_targets = batch_targets.to(device,non_blocking=True)

                norm_batch = helpers_data.compute_norm_eval(dataset_or_img=batch_inputs,data_normalisation=data_normalisation)
                normed_batch_inputs = helpers_data.normalize_eval(dataset_or_img=batch_inputs,data_normalisation=data_normalisation,
                                                           norm=norm_batch,params=model.params,to_torch=False)
                normed_batch_targets = helpers_data.normalize_eval(dataset_or_img=batch_targets,data_normalisation=data_normalisation,
                                                           norm=norm_batch,params=model.params,to_torch=False)

                fakePVfree = model.forward(normed_batch_inputs)
                fakePVfree_denormed = helpers_data.denormalize_eval(dataset_or_img=fakePVfree,data_normalisation=data_normalisation,
                                                                    norm=norm_batch,params=model.params,to_numpy=False)


                mean_norm = (torch.sum(torch.abs(batch_targets),dim = (1,2,3)) / batch_targets.shape[2] / batch_targets.shape[3])

                if do_NRMSE:
                    MSE = torch.sum((fakePVfree_denormed - batch_targets)**2, dim = (1,2,3)) / batch_targets.shape[2] / batch_targets.shape[3]
                    RMSE = torch.sqrt(MSE)
                    NRMSE = RMSE / mean_norm
                    list_NRMSE = torch.concat((list_NRMSE,NRMSE))

                if do_NMAE:
                    MAE = torch.sum(torch.abs(fakePVfree_denormed - batch_targets), dim=(1,2,3)) / batch_targets.shape[2] / batch_targets.shape[3]
                    NMAE = MAE / mean_norm
                    list_NMAE = torch.concat((list_NMAE,NMAE))

    if do_NRMSE:
        MNRMSE = torch.mean(list_NRMSE) / model.params['nb_gpu']
        if model.params['jean_zay']: dist.all_reduce(MNRMSE, op=dist.ReduceOp.SUM)
    if do_NMAE:
        MNMAE = torch.mean(list_NMAE)  / model.params['nb_gpu']
        if model.params['jean_zay']: dist.all_reduce(MNMAE, op=dist.ReduceOp.SUM)

    return MNRMSE, MNMAE