import itk
import torch
import numpy as np
import glob
import time
from torch.utils.data import Dataset,DataLoader


from . import helpers_data_parallelism, helpers

class CustomPVEProjectionsDataset(Dataset):
    def __init__(self, params, paths,filetype=None,merged=None,test=False):

        self.dataset_path = paths
        self.filetype = params["datatype"] if (filetype is None) else filetype
        self.merged = params["merged"] if (merged is None) else merged
        self.with_adj_angles = params["with_adj_angles"]
        self.noisy = (params['with_noise'])
        self.with_rec_fp = params['with_rec_fp']
        self.input_eq_angles = params['input_eq_angles']
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

        self.list_files=sorted(self.list_files)
        self.img_type = self.get_dtype(params['dtype'])

        first_img = self.read(filename=self.list_files[0])
        self.nb_pix_x,self.nb_pix_y = first_img.shape[1],first_img.shape[2]
        self.nb_projs_per_img = first_img.shape[0] if not self.merged else (int(first_img.shape[0]/3) if self.noisy else int(first_img.shape[0]/2))

        self.max_nb_data=params['max_nb_data']
        if (self.max_nb_data>0 and len(self.list_files)*self.nb_projs_per_img>self.max_nb_data):
            self.list_files=self.list_files[:int(self.max_nb_data/self.nb_projs_per_img)]

        if ('split_dataset' in params and params['split_dataset'] and not test):
            self.gpu_id, self.number_gpu = helpers_data_parallelism.get_gpu_id_nb_gpu(jean_zay=params['jean_zay'])
            self.list_files = list(np.array_split(self.list_files,self.number_gpu)[self.gpu_id])
        self.verbose=params['verbose']
        if self.verbose>1:
            print(f'First : {self.list_files[0]}')

        self.nb_src = len(self.list_files)

        self.list_transforms = params['data_augmentation']
        self.init_transforms()

        self.build_channels_id()
        self.build_merged_type_id()

        if self.store_dataset:
            self.build_numpy_dataset()
            del self.list_files

        self.len_dataset = self.cpu_dataset.shape[0] * self.nb_projs_per_img if self.store_dataset\
            else len(self.list_files) * self.nb_projs_per_img


    def read(self,filename, projs=None):
        if self.filetype in ['mha', 'mhd']:
            return itk.array_from_image(itk.imread(filename)).astype(dtype=self.img_type) if projs is None else\
                itk.array_from_image(itk.imread(filename))[projs,:,:].astype(dtype=self.img_type)
        elif self.filetype=='npy':
            return np.load(filename).astype(dtype=self.img_type) if projs is None else\
                np.load(filename)[projs,:,:].astype(dtype=self.img_type)

    def get_dtype(self,opt_dtype):
        if opt_dtype == 'float64':
            return np.float64
        elif opt_dtype == 'float32':
            return np.float32
        elif opt_dtype == 'float16':
            return np.float16
        elif opt_dtype == 'uint16':
            return np.uint16
        elif opt_dtype == 'uint64':
            return np.uint


    def build_numpy_dataset(self):
        if self.verbose>0:
            print(f'Loading data ...')
        t0 = time.time()

        self.nb_proj_type=3 if self.noisy else 2

        self.cpu_dataset = np.zeros((len(self.list_files), self.nb_proj_type,self.nb_projs_per_img,self.nb_pix_x,self.nb_pix_y),dtype=self.img_type)
        if self.verbose > 0:
            print(f'Shape of cpu_dataset : {self.cpu_dataset.shape}')
            print(f'Size of cpu_dataset : {(self.cpu_dataset.itemsize * self.cpu_dataset.size)/10**9} GB')

        for item_id,filename in enumerate(self.list_files):
            self.cpu_dataset[item_id, 0:self.nb_proj_type, :, :, :] = self.get_sinogram(filename=filename)


        self.cpu_dataset=torch.from_numpy(self.cpu_dataset).to('cpu')

        t1 = time.time()
        elapsed_time1 = t1 - t0
        if self.verbose > 0:
            print(self.cpu_dataset.shape)
            print(f'Done! in {elapsed_time1} s')

    def build_merged_type_id(self):
        if self.noisy:
            total_nb_of_projs = self.nb_projs_per_img*3
            cut1, cut2 = int(total_nb_of_projs / 3), int(2 * total_nb_of_projs / 3)

            d1 = np.arange(0, cut1)
            d2 = np.arange(cut1, cut2)
            d3 = np.arange(cut2,total_nb_of_projs)
            self.merged_type_id = np.concatenate((d1[None,:], d2[None,:], d3[None,:]), axis=0)
        else:
            total_nb_of_projs = self.nb_projs_per_img*2
            cut1 = int(total_nb_of_projs/2)
            d1 = np.arange(0,cut1)
            d2 = np.arange(cut1, total_nb_of_projs)
            self.merged_type_id = np.concatenate((d1[None, :], d2[None, :]), axis=0)

    def build_channels_id(self):
        # rotating channels id
        step = int(self.nb_projs_per_img / (self.input_eq_angles))
        self.channels_id = np.array([0])
        if self.with_adj_angles:
            adjacent_channels_id = np.array([(-1) % self.nb_projs_per_img, (1) % self.nb_projs_per_img])
            self.channels_id = np.concatenate((self.channels_id, adjacent_channels_id))

        equiditributed_channels_id = np.array([(k * step) % self.nb_projs_per_img for k in range(1, self.input_eq_angles)])
        self.channels_id = np.concatenate((self.channels_id, equiditributed_channels_id)) if len(
            equiditributed_channels_id) > 0 else self.channels_id


    def get_channels_id_i(self, proj_i):
        return (self.channels_id+proj_i)%120

    def get_sinogram(self,filename):
        return self.get_sinogram_merged(filename=filename) if self.merged else self.get_sinogram_not_merged(filename_PVE=filename)

    def get_sinogram_not_merged(self, filename_PVE):
        sinogram_PVE = self.read(filename=filename_PVE)

        filename_PVf = f'{filename_PVE[:-8]}_PVfree.{self.filetype}'
        sinogram_PVfree = self.read(filename=filename_PVf)

        if self.noisy:
            filename_noisy = f'{filename_PVE[:-8]}_PVE_noisy.{self.filetype}'
            sinogram_noisy = self.read(filename=filename_noisy)
            return np.stack((sinogram_noisy,sinogram_PVE,sinogram_PVfree), axis=0)
        else:
            return np.stack((sinogram_PVE,sinogram_PVfree),axis=0)

    def get_sinogram_merged(self, filename):
        return self.read(filename=filename, projs=self.merged_type_id)

    def init_transforms(self):
        self.transforms = []
        for trsfm in self.list_transforms:
            if trsfm=='noise':
                self.transforms.append(self.apply_noise)

    def apply_noise(self, input_sinogram):
        input = input_sinogram[1, :, :, :] if self.noisy else input_sinogram[0,:,:,:]
        input_sinogram[0,:,:,:] = np.random.poisson(lam=input, size=input.shape).astype(dtype=input.dtype)
        return input_sinogram


    def np_transforms(self, x):
        for trnsfm in self.transforms:
            x = trnsfm(x)
        return x

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, item_id):
        src_i = item_id % self.nb_src
        proj_i = item_id // self.nb_src
        channels_id_i = self.get_channels_id_i(proj_i=proj_i)
        if self.store_dataset:
            return (self.cpu_dataset[src_i,0,channels_id_i,:,:].float(),self.cpu_dataset[src_i,2,proj_i:proj_i+1,:,:].float())
        else:
            sinogram = self.get_sinogram(self.list_files[src_i])
            sinogram_input_channels = self.np_transforms(sinogram[:,channels_id_i,:,:])
            if self.with_rec_fp:
                rec_fp_filename = self.list_files[src_i].replace('_noisy_PVE_PVfree', '_rec_fp') if self.merged else self.list_files[src_i].replace('_PVE', '_rec_fp')
                rec_fp = self.read(rec_fp_filename, projs=np.array([proj_i]))
                x_inputs = np.concatenate((sinogram_input_channels[0,:,:,:], rec_fp),axis=0)
                return (torch.Tensor(x_inputs), torch.Tensor(sinogram_input_channels[2,0:1,:,:]))
            else:
                temp = torch.Tensor(self.np_transforms(sinogram_input_channels))
                return (temp[0, :, :, :], temp[2, 0:1, :, :])


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
                                  pin_memory=True,
                                  sampler=train_sampler)

    test_dataset = CustomPVEProjectionsDataset(params=params, paths=params['test_dataset_path'], test=True)
    test_batchsize = params['test_batchsize']

    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=test_batchsize,
                                  shuffle=False,
                                  num_workers=params['num_workers'],
                                  pin_memory=True,
                                  sampler=None)

    nb_training_data = len(train_dataloader.dataset)
    nb_testing_data = len(test_dataloader.dataset)
    if params['verbose']>0:
        print(f'Number of training data : {nb_training_data}')
        print(f'Number of testing data : {nb_testing_data}')
    params['nb_training_data'] = nb_training_data
    params['nb_testing_data'] = nb_testing_data
    params['nb_gpu'] = number_gpu
    params['training_mini_batchsize'] = training_batch_size_per_gpu
    params['test_mini_batchsize'] = test_batchsize
    params['norm'] = 'none'

    return train_dataloader, test_dataloader,params
