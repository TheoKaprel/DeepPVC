import numpy as np
import torch


def validation_errors(test_dataloader, model, do_NRMSE=True, do_NMAE=True):
    MNRMSE,std_NRMSE = 0,0
    MNMAE,std_NMAE = 0,0

    list_NRMSE = np.array([])
    list_NMAE = np.array([])

    with torch.no_grad():
        for test_it,batch in enumerate(test_dataloader):
            fakePVfree = model.forward(batch)

            batch_targets = batch[:,-1,0:1,:,:]
            print(batch_targets.shape)
            print(fakePVfree.shape)

            mean_norm = (torch.sum(torch.abs(batch_targets),dim = (1,2,3)) / batch.shape[2] / batch.shape[3])

            if do_NRMSE:
                MSE = torch.sum((fakePVfree - batch_targets)**2, dim = (1,2,3)) / batch.shape[2] / batch.shape[3]
                RMSE = torch.sqrt(MSE)
                NRMSE = RMSE / mean_norm
                list_NRMSE = np.concatenate((list_NRMSE,NRMSE.cpu().numpy()))

            if do_NMAE:
                MAE = torch.sum(torch.abs(fakePVfree - batch_targets), dim=(1,2,3)) / batch.shape[2] / batch.shape[3]
                NMAE = MAE / mean_norm
                list_NMAE = np.concatenate((list_NMAE, NMAE.cpu().numpy()))

    if do_NRMSE:
        MNRMSE = np.mean(list_NRMSE)
        std_NRMSE = np.std(list_NRMSE)
    if do_NMAE:
        MNMAE = np.mean(list_NMAE)
        std_NMAE = np.std(list_NMAE)


    return (MNRMSE,std_NRMSE), (MNMAE,std_NMAE)