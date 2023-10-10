import itk
import torch
import numpy as np
import glob
import time
import h5py
from torch.utils.data import Dataset,DataLoader


from . import helpers_data_parallelism, helpers,helpers_data

class BaseCustomPVEProjectionsDataset(Dataset):
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
        self.img_type = self.get_dtype(params['dtype'])
        self.verbose = params['verbose']
        self.list_transforms = params['data_augmentation']
        self.max_nb_data = params['max_nb_data']
        self.test = test
        self.params = params

        if "sino" in self.params:
            self.sino=True
            self.nb_sino = self.params['sino']
        else:
            self.sino=False

        if params['network'] == 'unet_denoiser_pvc':
            self.double_model = True
        else:
            self.double_model = False

        if self.filetype in ['mhd', 'mha', 'npy', 'pt']:
            self.init_mhd_mha_npy()
            self._getitem = self.get_item_mhd_mha_npy
        elif self.filetype=='h5':
            self.init_h5()
            self._getitem = self.get_item_h5

    def init_mhd_mha_npy(self):
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
        first_img = self.read(filename=self.list_files[0])
        self.nb_pix_x,self.nb_pix_y = first_img.shape[1],first_img.shape[2]
        self.nb_projs_per_img = first_img.shape[0] if not self.merged else (int(first_img.shape[0]/3) if self.noisy else int(first_img.shape[0]/2))
        # #sino
        # if self.params['sino']:
        #     self.nb_sino=self.params['sino']
        #     self.nb_projs_per_img = self.nb_pix_x
        # #end sino

        if (self.max_nb_data>0 and len(self.list_files)*self.nb_projs_per_img>self.max_nb_data):
            self.list_files=self.list_files[:int(self.max_nb_data/self.nb_projs_per_img)]

        if ('split_dataset' in self.params and self.params['split_dataset'] and not self.test):
            self.gpu_id, self.number_gpu = helpers_data_parallelism.get_gpu_id_nb_gpu(jean_zay=self.params['jean_zay'])
            self.list_files = list(np.array_split(self.list_files,self.number_gpu)[self.gpu_id])

        if self.verbose>1:
            print(f'First : {self.list_files[0]}')
        self.nb_src = len(self.list_files)

        self.init_transforms()
        self.build_channels_id()
        self.build_merged_type_id()

        if self.store_dataset:
            self.build_numpy_dataset()
            del self.list_files

        self.len_dataset = self.cpu_dataset.shape[0] * self.nb_projs_per_img if self.store_dataset\
            else len(self.list_files) * self.nb_projs_per_img

    def init_h5(self):
        self.datasetfn = self.dataset_path[0]
        self.dataseth5 = h5py.File(self.datasetfn, 'r')
        self.keys = sorted(list(self.dataseth5.keys()))

        first_data = self.dataseth5[self.keys[0]]['PVE_noisy']
        self.nb_projs_per_img,self.nb_pix_x,self.nb_pix_y = first_data.shape[0],first_data.shape[1],first_data.shape[2]
        if self.sino:
            self.nb_projs_per_img, self.nb_pix_x, self.nb_pix_y = first_data.shape[1], first_data.shape[0],first_data.shape[2]
            self.zero_padding_for_sino = np.zeros((2, self.nb_sino+2 if self.with_rec_fp else self.nb_sino+1, 4, self.nb_pix_y),dtype=np.float32) if (self.double_model and not self.test) else np.zeros((self.nb_sino+2 if self.with_rec_fp else self.nb_sino+1, 4, self.nb_pix_y),dtype=np.float32)
            self.zero_padding_for_sino_pvfree = np.zeros((1,4, self.nb_pix_y),dtype=np.float32)

        self.build_channels_id()

        if (self.max_nb_data>0 and len(self.keys)*self.nb_projs_per_img>self.max_nb_data):
            self.keys=self.keys[:int(self.max_nb_data/self.nb_projs_per_img)]

        if ('split_dataset' in  self.params and  self.params['split_dataset'] and not  self.test):
            self.gpu_id, self.number_gpu = helpers_data_parallelism.get_gpu_id_nb_gpu(jean_zay= self.params['jean_zay'])
            self.keys = list(np.array_split(self.keys,self.number_gpu)[self.gpu_id])

        if self.verbose>1:
            print(f'First : {self.keys[0]}')
        self.nb_src = len(self.keys)

        self.init_transforms()
        self.build_channels_id()
        
        self.pad = torch.nn.ConstantPad2d((0, 0, 4, 4), 0)
        self.len_dataset = self.nb_src * self.nb_projs_per_img if not self.params['full_sino'] else self.nb_src


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

    def build_channels_id(self):
        # rotating channels id
        step = int(self.nb_projs_per_img / (self.input_eq_angles))
        self.channels_id = np.array([0])
        if self.with_adj_angles:
            adjacent_channels_id = np.array([(-1) % self.nb_projs_per_img, (1) % self.nb_projs_per_img])
            #sino
            if self.sino:
                adjacent_channels_id = np.array([(-k) % self.nb_projs_per_img for k in range(1,self.nb_sino//2+1)]+
                                                [(k) % self.nb_projs_per_img for k in range(1,self.nb_sino//2+1)])
            # end sino
            self.channels_id = np.concatenate((self.channels_id, adjacent_channels_id))

        equiditributed_channels_id = np.array([(k * step) % self.nb_projs_per_img for k in range(1, self.input_eq_angles)])
        self.channels_id = np.concatenate((self.channels_id, equiditributed_channels_id)) if len(
            equiditributed_channels_id) > 0 else self.channels_id

    def get_channels_id_i(self, proj_i):
        return (self.channels_id+proj_i)%self.nb_projs_per_img

    def init_transforms(self):
        self.transforms = []
        for trsfm in self.list_transforms:
            if (trsfm=='noise' and self.test==False):
                self.transforms.append(self.apply_noise)
        if self.verbose>=0:
            print(f'transforms : {self.transforms}')

    def apply_noise(self, input_sinogram):
        input = input_sinogram[1, :, :, :] if self.noisy else input_sinogram[0,:,:,:]
        input_sinogram[0,:,:,:] = np.random.poisson(lam=input, size=input.shape).astype(dtype=input.dtype)
        return input_sinogram

    def np_transforms(self, x):
        for trnsfm in self.transforms:
            x= trnsfm(x)
        return x

    def read(self,filename, projs=None):
        if self.filetype in ['mha', 'mhd']:
            return itk.array_from_image(itk.imread(filename)) if projs is None else\
                itk.array_from_image(itk.imread(filename))[projs,:,:]
        elif self.filetype=='npy':
            return np.load(filename)if projs is None else\
                np.load(filename)[projs,:,:]
        elif self.filetype=='pt':
            return torch.load(filename) if projs is None else torch.load(filename)[projs,:,:]

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

    def __len__(self):
        return self.len_dataset

    def get_item_mhd_mha_npy(self, item):
        src_i = item % self.nb_src
        proj_i = item // self.nb_src
        channels_id_i = self.get_channels_id_i(proj_i=proj_i)
        if self.store_dataset:
            return (self.cpu_dataset[src_i,0,channels_id_i,:,:].float(),self.cpu_dataset[src_i,2,proj_i:proj_i+1,:,:].float())
        else:
            sinogram_input_channels = self.get_sinogram(self.list_files[src_i])[:,channels_id_i,:,:]
            # /!\ np_transforms are applied before an eventual rec_fp channel concatenation ...
            sinogram_input_channels = self.np_transforms(sinogram_input_channels)
            temp_input,temp_target = sinogram_input_channels[0, :, :, :], sinogram_input_channels[2, 0:1, :, :]
            if self.with_rec_fp:
                rec_fp_filename = self.list_files[src_i].replace('_noisy_PVE_PVfree', '_rec_fp') if self.merged else self.list_files[src_i].replace('_PVE', '_rec_fp')
                rec_fp = self.read(rec_fp_filename, projs=np.array([proj_i]))
                temp_input = np.concatenate((temp_input, rec_fp),axis=0)
            return temp_input,temp_target

    def get_item_h5(self, item):
        src_i = item % self.nb_src
        proj_i = item // self.nb_src
        channels = self.get_channels_id_i(proj_i=proj_i)
        id = np.argsort(channels)
        invid = np.argsort(id)
        with h5py.File(self.datasetfn, 'r') as f:
            data = f[self.keys[src_i]]

            if not self.params['full_sino']:
                if not self.sino:
                    data_PVE_noisy,data_PVfree = np.array(data['PVE_noisy'][channels[id],:,:],dtype=np.float32)[invid],np.array(data['PVfree'][proj_i:proj_i+1,:,:],dtype=np.float32)
                else:
                #sino
                    data_PVE_noisy, data_PVfree = np.array(data['PVE_noisy'][:,channels[id],:], dtype=np.float32).transpose((1,0,2))[invid], np.array(data['PVfree'][:,proj_i:proj_i+1,:], dtype=np.float32).transpose((1,0,2))
                # end sino
                if self.with_rec_fp:
                    if not self.sino:
                        rec_fp = np.array(data['rec_fp'][proj_i:proj_i+1,:,:],dtype=np.float32)
                    else:
                    #sino
                        rec_fp = np.array(data['rec_fp'][:,proj_i:proj_i+1,:], dtype = np.float32).transpose((1,0,2))
                    #end sino
                    data_PVE_noisy = np.concatenate((data_PVE_noisy, rec_fp), axis=0)

                if (self.double_model and not self.test):
                    if not self.sino:
                        data_PVE = np.array(data['PVE'][channels[id],:,:],dtype=np.float32)[invid]
                    else:
                    # sino
                       data_PVE = np.array(data['PVE'][:,channels[id],:], dtype=np.float32).transpose((1,0,2))[invid]
                    # end sino
                    if self.with_rec_fp:
                        data_PVE = np.concatenate((data_PVE,rec_fp), axis=0)
                    data_PVE_noisy = np.stack((data_PVE_noisy,data_PVE))

                if self.sino:
                    data_PVE_noisy = np.concatenate((self.zero_padding_for_sino, data_PVE_noisy, self.zero_padding_for_sino), axis=2 if (self.double_model and not self.test) else 1)
                    data_PVfree = np.concatenate((self.zero_padding_for_sino_pvfree, data_PVfree, self.zero_padding_for_sino_pvfree), axis=1)
            else:
                data_PVE_noisy, data_PVfree = np.array(data['PVE_noisy'], dtype=np.float32), np.array(data['PVfree'],dtype=np.float32)
                data_PVE = np.array(data['PVE'], dtype=np.float32)


                if self.with_rec_fp:
                    data_rec_fp = np.array(data['rec_fp'], dtype=np.float32) # 120,256,256
                    data_PVE_noisy = np.concatenate((data_PVE_noisy, data_rec_fp), axis=0) # 240, 256, 256
                    data_PVE = np.concatenate((data_PVE, data_rec_fp), axis=0) # 240, 256, 256

                data_PVE_noisy = np.stack((data_PVE_noisy,data_PVE), axis=0) # 2, 240, 256, 256


                if self.sino:
                    data_PVE_noisy,data_PVfree = data_PVE_noisy.transpose((0, 2, 1, 3)), data_PVfree.transpose((1,0,2)) # (2, 256, 120, 256), # (256, 120, 256)

                    data_PVE_noisy = self.pad(torch.from_numpy(data_PVE_noisy))
                    data_PVfree = self.pad(torch.from_numpy(data_PVfree[None,:,:,:]))[0,:,:,:]

            return data_PVE_noisy,data_PVfree

    def __getitem__(self, item):
        return self._getitem(item)

def load_data(params):
    jean_zay=params['jean_zay']

    train_dataset = BaseCustomPVEProjectionsDataset(params=params, paths=params['dataset_path'],test=False)

    training_batchsize = params['training_batchsize']
    split_dataset = params['split_dataset']
    train_sampler, shuffle, pin_memory,number_gpu = helpers_data_parallelism.get_dataloader_params(dataset=train_dataset,
                                                                                                            jean_zay=jean_zay,
                                                                                                            split_dataset=split_dataset)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=training_batchsize,
                                  shuffle=shuffle,
                                  num_workers=params['num_workers'],
                                  pin_memory=True,
                                  sampler=train_sampler)

    test_dataset = BaseCustomPVEProjectionsDataset(params=params, paths=params['test_dataset_path'],test=True)
    test_batchsize = params['test_batchsize']

    if params['jean_zay']:
        import idr_torch
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,
                                                                  num_replicas=idr_torch.size,
                                                                  rank=idr_torch.rank,
                                                                  shuffle=False)
    else:
        test_sampler = None

    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=test_batchsize,
                                  shuffle=False,
                                  num_workers=4,
                                  pin_memory=True,
                                  sampler=test_sampler)

    if "validation_ref_type" in params:
        validation_dataset = helpers_data.load_image(filename=params["validation_ref_type"][0],
                                                     is_ref=True, type=params["validation_ref_type"][1],
                                                     params=params)

        if params['jean_zay']:
            import idr_torch
            val_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset,
                                                                           num_replicas=idr_torch.size,
                                                                           rank=idr_torch.rank,
                                                                           shuffle=False)
        else:
            val_sampler = None

        val_batchsize = 30

        validation_dataloader = DataLoader(dataset=validation_dataset,
                                           batch_size=val_batchsize,
                                           shuffle=False,
                                           num_workers=4,
                                           pin_memory=True,
                                           sampler=val_sampler)
    else:
        validation_dataloader = None

    nb_training_data = len(train_dataloader.dataset)
    nb_testing_data = len(test_dataloader.dataset)
    if params['verbose']>0:
        print(f'Number of training data : {nb_training_data}')
        print(f'Number of testing data : {nb_testing_data}')
    params['nb_training_data'] = nb_training_data
    params['nb_testing_data'] = nb_testing_data
    params['nb_gpu'] = number_gpu
    params['training_mini_batchsize'] = training_batchsize
    params['test_mini_batchsize'] = test_batchsize
    params['norm'] = 'none'

    return train_dataloader, test_dataloader,validation_dataloader,params
