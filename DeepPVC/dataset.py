import itk
import torch
import numpy as np
import glob

from . import helpers_data, helpers

def construct_dataset_from_path(dataset_path):
    print(f'Loading data from {dataset_path} ...')
    first = True
    dataset = None
    for filename_ in glob.glob(f'{dataset_path}/?????.mhd'): # selects files having exactly 5 characters before the .mhd
        filename_PVE = f'{filename_[:-4]}_PVE.mhd'
        img_PVE = itk.array_from_image(itk.imread(filename_PVE))

        filename_PVf = f'{filename_[:-4]}_PVfree.mhd'
        img_PVf = itk.array_from_image(itk.imread(filename_PVf))

        if min((np.max(img_PVE), np.max(img_PVf)))>0:
            cat_PVf_PVE = np.concatenate((img_PVE,img_PVf), axis =0)
            cat_PVf_PVE = np.expand_dims(cat_PVf_PVE,axis = 0)

            if first:
                dataset = cat_PVf_PVE
                first = False
            else:
                dataset = np.concatenate((dataset,cat_PVf_PVE),0)
    print('Done!')
    return dataset



def load_data(params):
    '''
    - Loads the dataset from a folder containing source (ref.mhd), projPVE (ref_PVE.mhd) and projPVfree (ref_PVfree.mhd)
    - Converts datas in ITK images and then in numpy arrays concatenated in an array (dataset) of shape (size_dataset, 2, 128,128)
    with dimension 1 contains in position   [0] PVE projection
                                            [1] PVfree projection
    - Normalizes data according to the maximum of each (128,128) image  --> [0,1] images
    - Converts the numpy array into torch tensor
    - Devides the dataset into training/test dataset
    - return the associated DataLoaders
    '''

    dataset_path = params['dataset_path']

    test_dataset_path = params['test_dataset_path']
    training_batchsize = params['training_batchsize']
    testing_batchsize = params['test_batchsize']

    normalisation = params['data_normalisation']
    device = helpers.get_auto_device(params['device'])

    dataset_is_set = False
    for path in dataset_path:
        tmp_dataset = construct_dataset_from_path(dataset_path=path)
        if dataset_is_set:
            dataset = np.concatenate((dataset, tmp_dataset), axis=0)
        else:
            dataset = tmp_dataset
            dataset_is_set = True



    norm = helpers_data.compute_norm(dataset, normalisation)
    normalized_train_dataset = helpers_data.normalize(dataset_or_img=dataset,normtype=normalisation, norm = norm, to_torch=True, device=device)
    params['norm'] = norm


    test_dataset_is_set = False
    for path in test_dataset_path:
        tmp_dataset = construct_dataset_from_path(dataset_path=path)
        if test_dataset_is_set:
            test_dataset = np.concatenate((test_dataset, tmp_dataset), axis=0)
        else:
            test_dataset = tmp_dataset
            test_dataset_is_set = True

    normalized_test_dataset = helpers_data.normalize(dataset_or_img=test_dataset, normtype=normalisation, norm = norm, to_torch=True, device=device)



    nb_training_data = len(normalized_train_dataset)
    nb_testing_data = len(normalized_test_dataset)
    print(f'Number of training data : {nb_training_data}')
    print(f'Number of testing data : {nb_testing_data}')
    params['nb_training_data'] = nb_training_data
    params['nb_testing_data'] = nb_testing_data


    train_dataloader = torch.utils.data.DataLoader(normalized_train_dataset, batch_size=training_batchsize, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(normalized_test_dataset, batch_size=testing_batchsize, shuffle=True)

    return train_dataloader,test_dataloader,params