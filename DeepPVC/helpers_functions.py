import numpy as np
import torch

from . import helpers_data
from torch.cuda.amp import autocast
import torch.distributed as dist

def validation_errors(test_dataloader, model, do_NRMSE=True, do_NMAE=True):
    data_normalisation = model.params['data_normalisation']
    device = model.device
    nb_testing_data = len(test_dataloader.dataset)
    MSE = torch.Tensor([0.]).to(device)
    MAE = torch.Tensor([0.]).to(device)

    for test_it,(batch_inputs,batch_targets) in enumerate(test_dataloader):
        batch_inputs = tuple([input_i.to(device, non_blocking=True) for input_i in batch_inputs])
        batch_targets = tuple([target_i.to(device, non_blocking=True) for target_i in batch_targets])

        ground_truth=batch_targets[1]

        with torch.no_grad():
            norm_batch = helpers_data.compute_norm_eval(dataset_or_img=batch_inputs,data_normalisation=data_normalisation)
            batch_inputs = helpers_data.normalize_eval(dataset_or_img=batch_inputs,data_normalisation=data_normalisation,
                                                       norm=norm_batch,params=model.params,to_torch=False)

            fakePVfree = model.forward(batch_inputs)
            fakePVfree = helpers_data.denormalize_eval(dataset_or_img=fakePVfree,data_normalisation=data_normalisation,
                                                                norm=norm_batch,params=model.params,to_numpy=False)


        if do_NRMSE:
            MSE_batch = torch.mean((fakePVfree-ground_truth)**2)
            MSE += MSE_batch.item()*batch_inputs[0].size(0)/nb_testing_data
        if do_NMAE:
            MAE_batch = torch.mean(torch.abs(fakePVfree - ground_truth))
            MAE += MAE_batch.item()*batch_inputs[0].size(0)/nb_testing_data

    if do_NRMSE:
        if model.params['jean_zay']:
            dist.all_reduce(MSE, op=dist.ReduceOp.SUM)
    if do_NMAE:
        if model.params['jean_zay']:
            dist.all_reduce(MAE, op=dist.ReduceOp.SUM)
    return MSE, MAE