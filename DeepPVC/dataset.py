import itk
import torch
import numpy as np
import glob
import time
from torch.utils.data import Dataset,DataLoader


from . import helpers_data,helpers_data_parallelism, helpers

class CustomPVEProjectionsDataset(Dataset):
    def __init__(self, params, paths,filetype=None,merged=None,test=False):

        self.dataset_path = paths
        self.filetype = params["datatype"] if (filetype is None) else filetype
        self.merged = params["merged"] if (merged is None) else merged
        self.with_adj_angles = params["with_adj_angles"]
        self.noisy = (params['with_noise'])
        self.input_channels = params['input_channels']
        self.data_normalisation = params['data_normalisation']
        self.device = helpers.get_auto_device(params['device'])
        self.store_dataset = params['store_dataset']

        self.list_files = []
        for path in self.dataset_path:
            if self.merged:
                if self.noisy:
                    self.list_files.extend(glob.glob(f'{path}/?????_noisy_PVE_PVfree.{self.filetype}'))
                else:
                    self.list_files.extend(glob.glob(f'{path}/?????_PVE_PVfree.{self.filetype}'))
            else:
                self.list_files.extend(glob.glob(f'{path}/?????_PVE.{self.filetype}'))

        first_img = self.read(filename=self.list_files[0])
        self.nb_pix_x,self.nb_pix_y = first_img.shape[1],first_img.shape[2]
        self.nb_projs_per_img = first_img.shape[0] if not self.merged else (int(first_img.shape[0]/3) if self.noisy else int(first_img.shape[0]/2))
        self.img_type=first_img.dtype

        self.max_nb_data=params['max_nb_data']
        if (self.max_nb_data>0 and len(self.list_files)*self.nb_projs_per_img>self.max_nb_data):
            self.list_files=self.list_files[:int(self.max_nb_data/self.nb_projs_per_img)]

        if ('split_dataset' in params and params['split_dataset'] and not test):
            self.gpu_id, self.number_gpu = helpers_data_parallelism.get_gpu_id_nb_gpu(jean_zay=params['jean_zay'])
            self.list_files = list(np.array_split(self.list_files,self.number_gpu)[self.gpu_id])

        self.nb_src = len(self.list_files)

        if self.store_dataset:
            self.build_numpy_dataset()
            del self.list_files

        self.len_dataset = self.numpy_cpu_dataset.shape[0] * self.nb_projs_per_img if self.store_dataset\
            else len(self.list_files) * self.nb_projs_per_img


    def read(self,filename):
        if self.filetype in ['mha', 'mhd']:
            return itk.array_from_image(itk.imread(filename))
        elif self.filetype=='npy':
            return np.load(filename)


    def build_numpy_dataset(self):
        print(f'Loading data ...')
        t0 = time.time()

        self.nb_proj_type=3 if self.noisy else 2

        self.numpy_cpu_dataset = np.zeros((len(self.list_files), self.nb_proj_type,self.nb_projs_per_img,self.nb_pix_x,self.nb_pix_y),dtype=self.img_type)

        print(f'Size of numpy_cpu_dataset : {(self.numpy_cpu_dataset.itemsize * self.numpy_cpu_dataset.size)/10**9} GB')

        for item_id,filename in enumerate(self.list_files):
            self.numpy_cpu_dataset[item_id, 0:self.nb_proj_type, :, :, :] = self.get_sinogram(filename=filename)

        # conversion if the array is type uint16 because impossible for torch to convert it (why?)
        if self.numpy_cpu_dataset.dtype==np.uint16:
            self.numpy_cpu_dataset=self.numpy_cpu_dataset.astype(np.int16)

        t1 = time.time()
        elapsed_time1 = t1 - t0
        print(self.numpy_cpu_dataset.shape)
        print(f'Done! in {elapsed_time1} s')


    def get_sinogram(self,filename):
        return self.get_sinogram_merged(filename=filename) if self.merged else self.get_sinogram_not_merged(filename_PVE=filename)

    def get_sinogram_not_merged(self, filename_PVE):
        sinogram_PVE = self.read(filename=filename_PVE)[None,:,:,:]

        filename_PVf = f'{filename_PVE[:-8]}_PVfree.{self.filetype}'
        sinogram_PVfree = self.read(filename=filename_PVf)[None,:,:,:]

        if self.noisy:
            filename_noisy = f'{filename_PVE[:-8]}_PVE_noisy.{self.filetype}'
            sinogram_noisy = self.read(filename=filename_noisy)[None,:,:,:]
            return np.concatenate((sinogram_noisy,sinogram_PVE,sinogram_PVfree), axis=0)
        else:
            return np.concatenate((sinogram_PVE,sinogram_PVfree),axis=0)

    def get_sinogram_merged(self, filename):
        projs_merged = self.read(filename=filename)
        total_nb_of_projs = projs_merged.shape[0]
        if self.noisy:
            cut1,cut2 = int(total_nb_of_projs/3),int(2*total_nb_of_projs/3)
            sinogram_noisy = projs_merged[None,0:cut1,:,:]
            sinogram_PVE = projs_merged[None,cut1:cut2,:,:]
            sinogram_PVfree = projs_merged[None,cut2:total_nb_of_projs,:,:]
            return np.concatenate((sinogram_noisy, sinogram_PVE, sinogram_PVfree), axis=0)
        else:
            cut1 = int(total_nb_of_projs/2)
            sinogram_PVE = projs_merged[None,0:cut1,:,:]
            sinogram_PVfree = projs_merged[None,cut1:total_nb_of_projs,:,:]
            return np.concatenate((sinogram_PVE, sinogram_PVfree), axis=0)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, item_id):
        if self.store_dataset:
            img_channels = helpers_data.load_img_channels(img_array=self.numpy_cpu_dataset[item_id%self.nb_src,:,:,:,:],
                                                          proj_i=item_id//self.nb_src,
                                                          nb_channels=self.input_channels,with_adj_angles=self.with_adj_angles)
        else:
            img_channels = helpers_data.load_img_channels(img_array=self.get_sinogram(self.list_files[item_id%self.nb_src]),
                                                          proj_i=item_id//self.nb_src,
                                                          nb_channels=self.input_channels,with_adj_angles=self.with_adj_angles)

            if img_channels.dtype == np.uint16:
                img_channels = img_channels.astype(np.int16)

        return torch.from_numpy(img_channels)


def load_data(params):
    jean_zay=params['jean_zay']

    train_dataset = CustomPVEProjectionsDataset(params=params, paths=params['dataset_path'],test=False)
    training_batchsize = params['training_batchsize']
    split_dataset = params['split_dataset']
    train_sampler, shuffle, training_batch_size_per_gpu, pin_memory,number_gpu = helpers_data_parallelism.get_dataloader_params(dataset=train_dataset,
                                                                                                            batch_size=training_batchsize,
                                                                                                            jean_zay=jean_zay,
                                                                                                            split_dataset=split_dataset)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=training_batch_size_per_gpu,
                                  shuffle=shuffle,
                                  num_workers=params['num_workers'],
                                  pin_memory=pin_memory,
                                  sampler=train_sampler)

    test_dataset = CustomPVEProjectionsDataset(params=params, paths=params['test_dataset_path'], test=True)
    test_batchsize = params['test_batchsize']

    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=test_batchsize,
                                  shuffle=False,
                                  num_workers=0,
                                  pin_memory=True,
                                  sampler=None)

    nb_training_data = len(train_dataloader.dataset)
    nb_testing_data = len(test_dataloader.dataset)
    print(f'Number of training data : {nb_training_data}')
    print(f'Number of testing data : {nb_testing_data}')
    params['nb_training_data'] = nb_training_data
    params['nb_testing_data'] = nb_testing_data
    params['nb_gpu'] = number_gpu
    params['training_mini_batchsize'] = training_batch_size_per_gpu
    params['test_mini_batchsize'] = test_batchsize
    params['norm'] = 'none'

    return train_dataloader, test_dataloader,params
