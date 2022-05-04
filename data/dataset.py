import itk
import torch
import numpy as np
import glob



def load_data(dataset_path, batchsize,prct_train):

    # dataset = np.empty((1,2,128,128))
    first = True
    for filename_ in glob.glob(f'{dataset_path}/?????.mhd'):
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


    mean = np.mean(dataset,axis=(0,2,3))
    std = np.std(dataset,axis=(0,2,3))
    mean_std = np.concatenate((mean[None,:], std[None,:]), axis=0)
    # dataset = (dataset - mean[None, :,None, None])/std[None, :,None, None]


    dataset = torch.from_numpy(dataset)

    # Division of dataset into train_dataset and test_dataset
    train_size = int(prct_train* len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

    return train_dataloader,test_dataloader