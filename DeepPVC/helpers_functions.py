import torch

from . import helpers_data





def validation_errors(dataset_loader, model, do_NRMSE=True, do_NMAE=True):
    params = model.params
    MNRMSE = 0
    MNMAE = 0
    N = 0
    with torch.no_grad():
        for test_it, batch in enumerate(dataset_loader):


            fakePVfree = model.forward(batch)

            denormalized_target = helpers_data.denormalize(batch[:,2,:,:,:],
                                                           normtype=params['data_normalisation'], norm=params['norm'],
                                                           to_numpy=False)
            denormalized_output = helpers_data.denormalize(fakePVfree, normtype=params['data_normalisation'],
                                                           norm=params['norm'], to_numpy=False)

            mean_norm = (torch.sum(torch.abs(denormalized_target), dim=(1,2,3)) / denormalized_output.shape[2] / denormalized_output.shape[3])

            if do_NRMSE:
                MSE = torch.sum((denormalized_output - denormalized_target)**2, dim = (1,2,3)) / denormalized_output.shape[2] / denormalized_output.shape[3]
                RMSE = torch.sqrt(MSE)
                NRMSE = RMSE / mean_norm
                MNRMSE += torch.sum(NRMSE).item()

            if do_NMAE:
                MAE = torch.sum(torch.abs(denormalized_output - denormalized_target), dim=(1,2,3)) / denormalized_output.shape[2] / denormalized_output.shape[3]
                NMAE = MAE / mean_norm
                MNMAE += torch.sum(NMAE).item()

            N += denormalized_output.shape[0]

    if do_NRMSE:
        MNRMSE = MNRMSE / N
    if do_NMAE:
        MNMAE = MNMAE / N

    return MNRMSE, MNMAE