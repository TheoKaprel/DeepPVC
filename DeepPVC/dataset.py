import itk
import torch
import numpy as np
import glob
import time
from torch.utils.data import Dataset,DataLoader

from . import helpers_data, helpers

def construct_dataset_from_path(dataset_path,datatype, nb_channels=1, noisy = False):
    print(f'Loading data from {dataset_path} ...')
    t0 = time.time()


    list_files = glob.glob(f'{dataset_path}/?????_PVE.{datatype}')
    N = len(list_files)
    if noisy:
        dataset = np.zeros((N, 3, nb_channels, 128, 128))
    else:
        dataset = np.zeros((N, 2, nb_channels, 128, 128))

    for (i,filename_PVE) in enumerate(list_files): # selects files having exactly 5 characters before the .mhd

        if noisy:
            filename_noisy = f'{filename_PVE[:-8]}_PVE_noisy.{datatype}'
            img_noisy = itk.array_from_image(itk.imread(filename_noisy))
            dataset[i,0,:,:,:] = img_noisy[0:nb_channels,:,:]
            next_input = 1
        else:
            next_input = 0

        img_PVE = itk.array_from_image(itk.imread(filename_PVE))

        dataset[i,next_input,:,:,:] = img_PVE[0:nb_channels,:,:]

        filename_PVf = f'{filename_PVE[:-8]}_PVfree.{datatype}'
        img_PVf = itk.array_from_image(itk.imread(filename_PVf))
        dataset[i,next_input+1,:,:,:] = img_PVf[0:nb_channels,:,:]


    t1 = time.time()
    elapsed_time1 = t1-t0
    print(f'Done! in {elapsed_time1} s')
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
    datatype = params["datatype"]
    test_dataset_path = params['test_dataset_path']
    training_batchsize = params['training_batchsize']

    noisy = (params['with_noise'])

    input_channels = params['input_channels']

    normalisation = params['data_normalisation']
    device = helpers.get_auto_device(params['device'])

    dataset_is_set = False
    for path in dataset_path:
        tmp_dataset = construct_dataset_from_path(dataset_path=path,datatype=datatype, nb_channels=input_channels, noisy=noisy)
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
        tmp_dataset = construct_dataset_from_path(dataset_path=path,datatype=datatype, nb_channels=input_channels, noisy=noisy)
        if test_dataset_is_set:
            test_dataset = np.concatenate((test_dataset, tmp_dataset), axis=0)
        else:
            test_dataset = tmp_dataset
            test_dataset_is_set = True

    nb_training_data = normalized_train_dataset.shape[0]
    nb_testing_data = test_dataset.shape[0]
    print(f'Number of training data : {nb_training_data}')
    print(f'Number of testing data : {nb_testing_data}')
    params['nb_training_data'] = nb_training_data
    params['nb_testing_data'] = nb_testing_data

    train_dataloader = DataLoader(normalized_train_dataset, batch_size=training_batchsize, shuffle=True)

    return train_dataloader,test_dataset,params




def load_test_data(datatype, params, from_folder=False,from_file=False, is_ref=False):
    noisy = (params['with_noise'])

    input_channels = params['input_channels']

    if from_folder!=False:
        test_dataset = construct_dataset_from_path(dataset_path=from_folder,datatype=datatype, nb_channels=input_channels, noisy=noisy)
    if from_file!=False:
        test_dataset = helpers_data.load_image(filename=from_file,is_ref=is_ref, type=datatype,noisy=noisy,nb_channels=input_channels)

    return test_dataset


class CustomPVEProjectionsDataset(Dataset):
    def __init__(self, params, paths):
        self.dataset_path = paths
        self.datatype = params["datatype"]
        self.noisy = (params['with_noise'])
        self.input_channels = params['input_channels']
        self.device = helpers.get_auto_device(params['device'])

        self.list_files = []
        for path in self.dataset_path:
            self.list_files.extend(glob.glob(f'{path}/?????_PVE.{self.datatype}'))

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, item_id):
        filename_PVE = self.list_files[item_id]

        if self.noisy:
            projections = torch.zeros((3, self.input_channels, 128, 128), device = self.device  )
            filename_noisy = f'{filename_PVE[:-8]}_PVE_noisy.{self.datatype}'
            img_noisy = torch.tensor(itk.array_from_image(itk.imread(filename_noisy)),device = self.device)
            projections[0,:,:,:] = img_noisy[0:self.input_channels,:,:]
            next_input = 1
        else:
            projections = torch.zeros((2, self.input_channels, 128, 128))
            next_input = 0

        img_PVE = torch.tensor(itk.array_from_image(itk.imread(filename_PVE)), device = self.device)

        projections[next_input,:,:,:] = img_PVE[0:self.input_channels,:,:]

        filename_PVf = f'{filename_PVE[:-8]}_PVfree.{self.datatype}'
        img_PVf = torch.tensor(itk.array_from_image(itk.imread(filename_PVf)), device = self.device)
        projections[next_input+1,:,:,:] = img_PVf[0:self.input_channels,:,:]

        return projections

def load_data_v2(params):
    train_dataset = CustomPVEProjectionsDataset(params=params, paths=params['dataset_path'])
    training_batchsize = params['training_batchsize']
    train_dataloader = DataLoader(train_dataset, batch_size=training_batchsize, shuffle=True)

    test_dataset = CustomPVEProjectionsDataset(params=params, paths=params['test_dataset_path'])
    test_batchsize = params['test_batchsize']
    test_dataloader = DataLoader(test_dataset,batch_size=test_batchsize,shuffle=False)

    nb_training_data = len(train_dataloader.dataset)
    nb_testing_data = len(test_dataloader.dataset)
    print(f'Number of training data : {nb_training_data}')
    print(f'Number of testing data : {nb_testing_data}')
    params['nb_training_data'] = nb_training_data
    params['nb_testing_data'] = nb_testing_data
    params['norm'] = None

    return train_dataloader, test_dataloader,params
