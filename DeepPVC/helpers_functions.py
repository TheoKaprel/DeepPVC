import torch

from . import helpers_data





def mean_square_error(dataset_loader, model):
    params = model.params
    MSE = 0
    with torch.no_grad():
        for test_it, batch in enumerate(dataset_loader):
            fakePVfree = model.forward(batch)

            denormalized_target = helpers_data.denormalize(model.truePVfree,
                                                           normtype=params['data_normalisation'], norm=params['norm'],
                                                           to_numpy=False)
            denormalized_output = helpers_data.denormalize(fakePVfree, normtype=params['data_normalisation'],
                                                           norm=params['norm'], to_numpy=False)

            norm = torch.sum(denormalized_target ** 2, dim=(1, 2, 3))

            MSE += torch.sum(torch.sum((denormalized_output - denormalized_target) ** 2, dim=(1, 2, 3)) / norm ) / params['nb_testing_data']

    return MSE