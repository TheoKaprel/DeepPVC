import itk
import torch
import numpy as np
import glob



def load_data(dataset_path, training_batchsize, testing_batchsize,prct_train, device):
    '''
    - Loads the dataset from a folder containing source (ref.mhd), projPVE (ref_PVE.mhd) and projPVfree (ref_PVfree.mhd)
    - Converts datas in ITK images and then in numpy arrays concatenated in an array (dataset) of shape (size_dataset, 2, 128,128)
    with dimension 1 contains in position   [0] PVE projection
                                            [1] PVfree projection
    - Normalizes data according to the maximum of each (128,128) image  --> [0,1] images
    - FIXME : no normalisation wrt mean/std
    - Converts the numpy array into torch tensor
    - Devides the dataset into training/test dataset
    - FIXME : no validation dataset yet
    - return the associated DataLoaders

    :param dataset_path: path to the data
    :param training_batchsize, testing_batchsize: number of couples (PVE,PVfree) per batch
    :param prct_train: percentage of the dataset going into the training dataset. The remaining goes into test dataset for now
    :return: train_dataloader,test_dataloader
    '''


    first = True
    for filename_ in glob.glob(f'{dataset_path}/?????.mhd'): # selects files having exactly 5 characters before the .mhd
        filename_PVE = f'{filename_[:-4]}_PVE.mhd'
        img_PVE = itk.array_from_image(itk.imread(filename_PVE))

        filename_PVf = f'{filename_[:-4]}_PVfree.mhd'
        img_PVf = itk.array_from_image(itk.imread(filename_PVf))

        cat_PVf_PVE = np.concatenate((img_PVE,img_PVf), axis =0)
        cat_PVf_PVE = np.expand_dims(cat_PVf_PVE,axis = 0)

        if np.max(cat_PVf_PVE)>0:
            if first:
                dataset = cat_PVf_PVE
                first = False
            else:
                dataset = np.concatenate((dataset,cat_PVf_PVE),0)


    # Ranges data between [0 , 1]
    data_max = np.max(dataset,axis=(2,3))[:,:,None,None]

    dataset = dataset/data_max


    # mean = np.mean(dataset,axis=(0,2,3))
    # std = np.std(dataset,axis=(0,2,3))
    # mean_std = np.concatenate((mean[None,:], std[None,:]), axis=0)
    # dataset = (dataset - mean[None, :,None, None])/std[None, :,None, None]


    dataset = torch.from_numpy(dataset).to(device)

    # Division of dataset into train_dataset and test_dataset
    train_size = int(prct_train* len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=training_batchsize, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=testing_batchsize, shuffle=True)

    return train_dataloader,test_dataloader