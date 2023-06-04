import numpy as np
import torch

from . import helpers_data
from torch.cuda.amp import autocast
import torch.distributed as dist

def validation_errors(test_dataloader, model, do_NRMSE=True, do_NMAE=True):
    data_normalisation = model.params['data_normalisation']
    device = model.device
    list_MSE,list_MAE = [],[]
    RMSE, MAE = 0,0

    with torch.no_grad():
        with autocast():
            for test_it,(batch_inputs,batch_targets) in enumerate(test_dataloader):
                batch_inputs = batch_inputs.to(device,non_blocking=True)
                batch_targets = batch_targets.to(device,non_blocking=True)

                norm_batch = helpers_data.compute_norm_eval(dataset_or_img=batch_inputs,data_normalisation=data_normalisation)
                normed_batch_inputs = helpers_data.normalize_eval(dataset_or_img=batch_inputs,data_normalisation=data_normalisation,
                                                           norm=norm_batch,params=model.params,to_torch=False)

                fakePVfree = model.forward(normed_batch_inputs)
                fakePVfree_denormed = helpers_data.denormalize_eval(dataset_or_img=fakePVfree,data_normalisation=data_normalisation,
                                                                    norm=norm_batch,params=model.params,to_numpy=False)


                if do_NRMSE:
                    MSE_batch = torch.mean((fakePVfree_denormed-batch_targets)**2)
                    list_MSE.append(MSE_batch.item())
                if do_NMAE:
                    MAE_batch = torch.mean(torch.abs(fakePVfree_denormed - batch_targets))
                    list_MAE.append(MAE_batch.item())

    if do_NRMSE:
        RMSE = np.sqrt(np.mean(list_MSE))
    if do_NMAE:
        MAE = np.mean(list_MAE)
    return RMSE, MAE