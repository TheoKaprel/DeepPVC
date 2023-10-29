import itk
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset,DataLoader


from . import helpers_data_parallelism, helpers,helpers_data

# no more: merged,noise,store_dataset
class BaseDataset(Dataset):
    def __init__(self, params, paths, filetype=None,merged=None,test=False):
        self.dataset_path = paths
        self.filetype = params["datatype"] if (filetype is None) else filetype
        self.with_rec_fp = params['with_rec_fp']
        self.data_normalisation = params['data_normalisation']
        self.device = helpers.get_auto_device(params['device'])
        self.img_type = self.get_dtype(params['dtype'])
        self.verbose = params['verbose']
        self.list_transforms = params['data_augmentation']
        self.max_nb_data = params['max_nb_data']
        self.test = test
        self.params = params
        self.double_model=True if params['network']=="unet_denoiser_pvc" else False

        self.dtype=self.get_dtype(params['dtype'])

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

    def apply_noise(self, input_sinogram):
        return np.random.poisson(lam=input_sinogram, size=input_sinogram.shape).astype(dtype=input_sinogram.dtype)


class ProjToProjDataset(BaseDataset):
    def __init__(self, params, paths, filetype=None, merged=None, test=False):
        super().__init__(params, paths, filetype, merged, test)
        self.with_adj_angles = params["with_adj_angles"]
        self.input_eq_angles = params['input_eq_angles']

        if "sino" in self.params:
            self.sino=True
            self.nb_sino = self.params['sino']
        else:
            self.sino=False

        if self.filetype in ['mhd', 'mha', 'npy', 'pt']:
            print(f"ERROR: {self.filetype} datasets are not handled any more...")
            exit(0)
        elif self.filetype=='h5':
            self.init_h5()

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

        self.build_channels_id()
        
        self.pad = torch.nn.ConstantPad2d((0, 0, 4, 4), 0)
        self.len_dataset = self.nb_src * self.nb_projs_per_img

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

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, item):
        return self.get_item_h5_projs(item)

    def get_item_h5_projs(self, item):
        src_i = item % self.nb_src
        proj_i = item // self.nb_src
        channels = self.get_channels_id_i(proj_i=proj_i)
        id = np.argsort(channels)
        invid = np.argsort(id)
        with h5py.File(self.datasetfn, 'r') as f:
            data = f[self.keys[src_i]]
            if not self.sino:
                data_PVE_noisy = np.array(data['PVE_noisy'][channels[id],:,:],dtype=np.float32)[invid]
                data_target = np.array(data['PVfree'][proj_i:proj_i+1,:,:],dtype=np.float32)
            else:
            #sino
                data_PVE_noisy, data_target = np.array(data['PVE_noisy'][:,channels[id],:], dtype=np.float32).transpose((1,0,2))[invid], np.array(data['PVfree'][:,proj_i:proj_i+1,:], dtype=np.float32).transpose((1,0,2))
                data_PVE_noisy = self.pad(torch.from_numpy(data_PVE_noisy[None, :, :, :]))[0, :, :, :]
                data_target = self.pad(torch.from_numpy(data_target[None, :, :, :]))[0, :, :, :]
            # end sino
            if self.with_rec_fp:
                if not self.sino:
                    rec_fp = np.array(data['rec_fp'][proj_i:proj_i+1,:,:],dtype=np.float32)
                else:
                #sino
                    rec_fp = np.array(data['rec_fp'][:,proj_i:proj_i+1,:], dtype = np.float32).transpose((1,0,2))
                    rec_fp = self.pad(torch.from_numpy(rec_fp[None, :, :, :]))[0, :, :, :]
                #end sino

            if (self.double_model and not self.test):
                if not self.sino:
                    data_PVE = np.array(data['PVE'][channels[id],:,:],dtype=np.float32)[invid]
                else:
                #sino
                    data_PVE = np.array(data['PVE'][:,channels[id],:], dtype=np.float32).transpose((1,0,2))[invid]
                    data_PVE = self.pad(torch.from_numpy(data_PVE[None, :, :, :]))[0, :, :, :]
                # end sino

                if "noise" in self.list_transforms:
                    data_PVE_noisy = self.apply_noise(data_PVE)

                data_inputs = (data_PVE_noisy,data_PVE)
            else:
                data_inputs = (data_PVE_noisy,)

            if self.with_rec_fp:
                data_inputs = data_inputs + (rec_fp,)

        return data_inputs, data_target


class SinoToSinoDataset(BaseDataset):
    def __init__(self, params, paths, filetype=None, merged=None, test=False):
        super().__init__(params, paths, filetype, merged, test)
        if params['pad']=="zero":
            self.pad = torch.nn.ConstantPad2d((0, 0, 0, 0, 4, 4), 0)
        else:
            self.pad = torch.nn.Identity()
        self.patches=params['patches']
        self.init_h5()

    def init_h5(self):
        self.datasetfn = self.dataset_path[0]
        self.dataseth5 = h5py.File(self.datasetfn, 'r')
        self.keys = sorted(list(self.dataseth5.keys()))

        first_data = np.array(self.dataseth5[self.keys[0]]['PVE_noisy'],dtype=self.dtype)
        self.nb_projs_per_img, self.nb_pix_x, self.nb_pix_y = first_data.shape[0], first_data.shape[1], \
                                                              first_data.shape[2]


        if self.patches:
            self.patch_size=(32,64,64)
            first_data=self.pad(torch.from_numpy(first_data))[None,:,:,:]
            first_data_patches = first_data.unfold(1, self.patch_size[0], self.patch_size[0]).unfold(2, self.patch_size[1], self.patch_size[1]).unfold(3, self.patch_size[2], self.patch_size[2])
            self.unfold_shape=first_data_patches.size()
            self.tile_shape=(self.unfold_shape[1],self.unfold_shape[2],self.unfold_shape[3])

        if (self.max_nb_data > 0 and len(self.keys)> self.max_nb_data):
            self.keys = self.keys[:int(self.max_nb_data)]

        if (self.params['split_dataset'] and not self.test):
            self.gpu_id, self.number_gpu = helpers_data_parallelism.get_gpu_id_nb_gpu(jean_zay=self.params['jean_zay'])
            self.keys = list(np.array_split(self.keys, self.number_gpu)[self.gpu_id])

        if self.verbose > 1:
            print(f'First : {self.keys[0]}')
            print(f'Shape : {first_data.shape}')
        self.nb_src = len(self.keys)

        self.len_dataset = self.nb_src
        if self.patches:
            self.len_dataset=self.len_dataset * self.tile_shape[0]*self.tile_shape[1]*self.tile_shape[2]

    def get_item_h5_full_sino(self, item):
        if self.patches:
            src_i=item%self.nb_src
            i,j,k=item%self.tile_shape[0],item%self.tile_shape[1],item%self.tile_shape[2]
        else:
            src_i=item

        with h5py.File(self.datasetfn, 'r') as f:
            data = f[self.keys[src_i]]
            data_target = np.array(data['PVfree'],dtype=self.dtype)
            data_PVE = np.array(data['PVE'], dtype=self.dtype)

            data_PVE_noisy = self.apply_noise(data_PVE) if 'noise' in self.list_transforms else np.array(data['PVE_noisy'], dtype=self.dtype)

            data_inputs = (data_PVE_noisy,data_PVE) # ( (120,256,256), (120,256,256) )

            if self.with_rec_fp:
                data_rec_fp = np.array(data['rec_fp'], dtype=self.dtype) # (120,256,256)
                data_inputs = data_inputs+(data_rec_fp,) # ( (120,256,256), (120,256,256) )
        #--------------------
        data_inputs=tuple([self.pad(torch.from_numpy(u)) for u in data_inputs])
        data_target = self.pad(torch.from_numpy(data_target))
        #--------------------

        if self.patches:
            # ----------------------------
            data_inputs = tuple([
                data[None,:,:,:]
                    .unfold(1, self.patch_size[0], self.patch_size[0])
                    .unfold(2, self.patch_size[1], self.patch_size[1])
                    .unfold(3, self.patch_size[2], self.patch_size[2])[0,i,j,k,:,:,:] for data in data_inputs])
            data_target=data_target[None,:,:,:].unfold(1, self.patch_size[0], self.patch_size[0]
                                                       ).unfold(2, self.patch_size[1], self.patch_size[1]
                                                                ).unfold(3, self.patch_size[2], self.patch_size[2]
                                                                         )[0,i,j,k,:,:,:]
            # ----------------------------

        return data_inputs,data_target

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, item):
        return self.get_item_h5_full_sino(item)

def get_dataset(params, paths,filetype=None,merged=None,test=False):
    if params['inputs']=="full_sino":
        return SinoToSinoDataset(params=params,paths=paths,filetype=filetype,merged=merged,test=test)
    elif params['inputs']=="projs":
        return ProjToProjDataset(params=params,paths=paths,filetype=filetype,merged=merged,test=test)


def load_data(params):
    jean_zay=params['jean_zay']

    train_dataset = get_dataset(params=params, paths=params['dataset_path'],test=False)

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

    test_dataset = get_dataset(params=params, paths=params['test_dataset_path'],test=True)
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
