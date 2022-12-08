import numpy as np
import torch

from . import helpers_data
from . import helpers




def validation_errors(test_dataset_numpy, model, do_NRMSE=True, do_NMAE=True):
    params = model.params
    MNRMSE,std_NRMSE = 0,0
    MNMAE,std_NMAE = 0,0

    list_NRMSE = np.array([])
    list_NMAE = np.array([])

    batch_size = model.params['test_batchsize']
    data_normalisation=params['data_normalisation']
    test_dataset_numpy_batch = np.array_split(ary=test_dataset_numpy,indices_or_sections=(test_dataset_numpy.shape[0]//batch_size + 1), axis=0)

    with torch.no_grad():
        for test_it, batch in enumerate(test_dataset_numpy_batch):

            norm = helpers_data.compute_norm_eval(dataset_or_img=batch,data_normalisation=data_normalisation)

            normalized_batch = helpers_data.normalize_eval(dataset_or_img=batch,data_normalisation=data_normalisation, norm=norm,params=model.params, to_torch=True)

            fakePVfree = model.forward(normalized_batch)

            denormalized_output = helpers_data.denormalize_eval(dataset_or_img=fakePVfree,data_normalisation=data_normalisation,norm=norm,params=model.params,to_numpy=True)

            batch_targets = batch[:,2,:,:,:]

            mean_norm = (np.sum(np.abs(batch_targets), axis=(1,2,3)) / batch.shape[2] / batch.shape[3])

            if do_NRMSE:
                MSE = np.sum((denormalized_output - batch_targets)**2, axis = (1,2,3)) / batch.shape[2] / batch.shape[3]
                RMSE = np.sqrt(MSE)
                NRMSE = RMSE / mean_norm
                list_NRMSE = np.concatenate((list_NRMSE,NRMSE))

            if do_NMAE:
                MAE = np.sum(np.abs(denormalized_output - batch_targets), axis=(1,2,3)) / batch.shape[2] / batch.shape[3]
                NMAE = MAE / mean_norm
                list_NMAE = np.concatenate((list_NMAE, NMAE))

    if do_NRMSE:
        MNRMSE = np.mean(list_NRMSE)
        std_NRMSE = np.std(list_NRMSE)
    if do_NMAE:
        MNMAE = np.mean(list_NMAE)
        std_NMAE = np.std(list_NMAE)


    return (MNRMSE,std_NRMSE), (MNMAE,std_NMAE)