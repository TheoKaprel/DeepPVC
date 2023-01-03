import itk
import torch
import numpy as np
import glob
import time
from torch.utils.data import Dataset,DataLoader

from . import helpers_data, helpers


class CustomPVEProjectionsDataset(Dataset):
    def __init__(self, params, paths, dataset_type):

        self.dataset_path = paths
        self.datatype = params["datatype"]
        self.noisy = (params['with_noise'])
        self.input_channels = params['input_channels']
        self.data_normalisation = params['data_normalisation']
        self.device = helpers.get_auto_device(params['device'])

        self.list_files = []
        for path in self.dataset_path:
            self.list_files.extend(glob.glob(f'{path}/?????_PVE.{self.datatype}'))

        first_img = itk.array_from_image(itk.imread(self.list_files[0]))
        self.nb_projs_per_img,self.nb_pix_x,self.nb_pix_y = first_img.shape[0],first_img.shape[1],first_img.shape[2]

        self.build_numpy_dataset()

        del self.list_files
        if (dataset_type=='train'):
            print('Dataset prenormalisation ...')
            self.norm = helpers_data.compute_norm(dataset=self.numpy_cpu_dataset,data_normalisation=self.data_normalisation)
            self.numpy_cpu_dataset = helpers_data.normalize(dataset_or_img=self.numpy_cpu_dataset,
                                                            normtype=self.data_normalisation,norm=self.norm,to_torch=False,
                                                            device='notneededbutitiscpu')

    def build_numpy_dataset(self):
        print(f'Loading data ...')
        t0 = time.time()
        if self.noisy:
            projs_per_item = 3 #todo: changer le nom de cette variable qui n'a aucun sens
        else:
            projs_per_item = 2

        self.numpy_cpu_dataset = np.zeros((len(self.list_files)*self.nb_projs_per_img, projs_per_item,self.input_channels,self.nb_pix_x,self.nb_pix_y))

        for item_id,filename_PVE in enumerate(self.list_files):
            if self.noisy:
                filename_noisy = f'{filename_PVE[:-8]}_PVE_noisy.{self.datatype}'
                img_noisy = itk.array_from_image(itk.imread(filename_noisy))
                self.numpy_cpu_dataset[item_id*self.nb_projs_per_img:(item_id+1)*self.nb_projs_per_img,0:1,:,:,:] = helpers_data.load_img_channels(img_array=img_noisy,nb_channels=self.input_channels)
                next_input = 1
            else:
                next_input = 0

            img_PVE = itk.array_from_image(itk.imread(filename_PVE))

            self.numpy_cpu_dataset[item_id*self.nb_projs_per_img:(item_id+1)*self.nb_projs_per_img,next_input:next_input+1,:,:,:] = helpers_data.load_img_channels(img_array=img_PVE,nb_channels=self.input_channels)

            filename_PVf = f'{filename_PVE[:-8]}_PVfree.{self.datatype}'
            img_PVf = itk.array_from_image(itk.imread(filename_PVf))
            self.numpy_cpu_dataset[item_id*self.nb_projs_per_img:(item_id+1)*self.nb_projs_per_img,next_input+1:next_input+2,:,:,:] = helpers_data.load_img_channels(img_array=img_PVf,nb_channels=self.input_channels)

        t1 = time.time()
        elapsed_time1 = t1 - t0
        print(self.numpy_cpu_dataset.shape)
        print(f'Done! in {elapsed_time1} s')


    def __len__(self):
        return self.numpy_cpu_dataset.shape[0]

    def __getitem__(self, item_id):
        return torch.tensor(self.numpy_cpu_dataset[item_id,:,:,:,:],device=self.device)


def load_data(params):
    train_dataset = CustomPVEProjectionsDataset(params=params, paths=params['dataset_path'], dataset_type='train')
    training_batchsize = params['training_batchsize']
    train_dataloader = DataLoader(train_dataset, batch_size=training_batchsize, shuffle=True)

    test_dataset = CustomPVEProjectionsDataset(params=params, paths=params['test_dataset_path'], dataset_type='validation')
    test_batchsize = params['test_batchsize']
    test_dataloader = DataLoader(test_dataset,batch_size=test_batchsize,shuffle=False)

    nb_training_data = len(train_dataloader.dataset)
    nb_testing_data = len(test_dataloader.dataset)
    print(f'Number of training data : {nb_training_data}')
    print(f'Number of testing data : {nb_testing_data}')
    params['nb_training_data'] = nb_training_data
    params['nb_testing_data'] = nb_testing_data
    params['norm'] = train_dataset.norm

    return train_dataloader, test_dataloader,params
