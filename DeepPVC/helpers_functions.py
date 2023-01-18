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
        for test_it,batch in enumerate(test_dataloader):
            with autocast():
                batch = batch.to(device)

                norm_batch = helpers_data.compute_norm_eval(dataset_or_img=batch,data_normalisation=data_normalisation)
                normed_batch = helpers_data.normalize_eval(dataset_or_img=batch,data_normalisation=data_normalisation,
                                                           norm=norm_batch,params=model.params,to_torch=False)

                fakePVfree = model.forward(normed_batch)
                fakePVfree_denormed = helpers_data.denormalize_eval(dataset_or_img=fakePVfree,data_normalisation=data_normalisation,
                                                                    norm=norm_batch,params=model.params,to_numpy=False)

                batch_targets = batch[:,-1,0:1,:,:]

                mean_norm = (torch.sum(torch.abs(batch_targets),dim = (1,2,3)) / batch.shape[2] / batch.shape[3])

                if do_NRMSE:
                    MSE = torch.sum((fakePVfree_denormed - batch_targets)**2, dim = (1,2,3)) / batch.shape[2] / batch.shape[3]
                    RMSE = torch.sqrt(MSE)
                    NRMSE = RMSE / mean_norm
                    # list_NRMSE = np.concatenate((list_NRMSE,NRMSE.cpu().numpy()))
                    list_NRMSE = torch.concat((list_NRMSE,NRMSE))

                if do_NMAE:
                    MAE = torch.sum(torch.abs(fakePVfree_denormed - batch_targets), dim=(1,2,3)) / batch.shape[2] / batch.shape[3]
                    NMAE = MAE / mean_norm
                    # list_NMAE = np.concatenate((list_NMAE, NMAE.cpu().numpy()))
                    list_NMAE = torch.concat((list_NMAE,NMAE))

    if do_NRMSE:
        MNRMSE = torch.mean(list_NRMSE) / model.params['nb_gpu']
        # std_NRMSE = torch.std(list_NRMSE)
        dist.all_reduce(MNRMSE, op=dist.ReduceOp.SUM)
        # dist.all_reduce(std_NRMSE, op=dist.ReduceOp.SUM)
    if do_NMAE:
        MNMAE = torch.mean(list_NMAE)  / model.params['nb_gpu']
        # std_NMAE = torch.std(list_NMAE)
        dist.all_reduce(MNMAE, op=dist.ReduceOp.SUM)
        # dist.all_reduce(std_NMAE, op=dist.ReduceOp.SUM)

    return MNRMSE, MNMAE