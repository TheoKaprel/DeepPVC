import itk
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset,DataLoader
from volumentations import *


from . import helpers_data_parallelism, helpers,helpers_data

# no more: merged,noise,store_dataset
class BaseDataset(Dataset):
    def __init__(self, params, paths, filetype=None,merged=None,test=False):
        self.dataset_path = paths
        self.filetype = params["datatype"] if (filetype is None) else filetype
        self.with_rec_fp = params['with_rec_fp']
        self.device = helpers.get_auto_device(params['device'])
        self.img_type = self.get_dtype(params['dtype'])
        self.verbose = params['verbose']
        self.list_transforms = params['data_augmentation']
        self.max_nb_data = params['max_nb_data']
        self.test = test
        self.params = params
        self.double_model=True if params['network']=="unet_denoiser_pvc" else False
        # self.double_model=True

        self.with_att = params['with_att']

        self.with_lesion=("lesion" in params["recon_loss"])

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
        if type(input_sinogram)==torch.Tensor:
            input_sinogram = input_sinogram.numpy()
        return np.random.poisson(lam=input_sinogram, size=input_sinogram.shape).astype(dtype=input_sinogram.dtype)


class ProjToProjDataset(BaseDataset):
    def __init__(self, params, paths, filetype=None, merged=None, test=False):
        super().__init__(params, paths, filetype, merged, test)
        if params['sino']:
            self.sino=True
            self.nb_sino = self.params['input_eq_angles']
        else:
            self.sino = False
            self.with_adj_angles = params["with_adj_angles"]
            self.input_eq_angles = params['input_eq_angles']
            self.nb_adj_angles = params['nb_adj_angles']

        if self.filetype in ['mhd', 'mha', 'npy', 'pt']:
            print(f"ERROR: {self.filetype} datasets are not handled any more...")
            exit(0)
        elif self.filetype=='h5':
            self.init_h5()

    def init_h5(self):
        self.datasetfn = self.dataset_path[0]
        self.dataseth5 = h5py.File(self.datasetfn, 'r')
        self.keys = sorted(list(self.dataseth5.keys()))

        first_data = self.dataseth5[self.keys[0]]['PVE_att_noisy']

        if self.sino:
            self.nb_projs_per_img, self.nb_pix_x, self.nb_pix_y = first_data.shape[1], first_data.shape[0],first_data.shape[2]
            self.zero_padding_for_sino = np.zeros((2, self.nb_sino+2 if self.with_rec_fp else self.nb_sino+1, 4, self.nb_pix_y),dtype=self.dtype) if (self.double_model and not self.test) else np.zeros((self.nb_sino+2 if self.with_rec_fp else self.nb_sino+1, 4, self.nb_pix_y),dtype=self.dtype)
            self.zero_padding_for_sino_pvfree = np.zeros((1,4, self.nb_pix_y),dtype=self.dtype)
        else:
            self.nb_projs_per_img, self.nb_pix_x, self.nb_pix_y = first_data.shape[0], first_data.shape[1], first_data.shape[2]

        if (self.max_nb_data>0 and len(self.keys)>self.max_nb_data):
            self.keys=self.keys[:int(self.max_nb_data)]

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
        self.channels_id = np.array([0])
        if self.sino:
            adjacent_channels_id = np.array([(-k) % self.nb_projs_per_img for k in range(1, self.nb_sino // 2 + 1)] +
                                            [(k) % self.nb_projs_per_img for k in range(1, self.nb_sino // 2 + 1)])
            self.channels_id = np.concatenate((self.channels_id, adjacent_channels_id))
        else:
            step = int(self.nb_projs_per_img / (self.input_eq_angles))

            # adj angles
            if self.with_adj_angles:
                # adjacent_channels_id = np.array([(-1) % self.nb_projs_per_img, (1) % self.nb_projs_per_img])

                adjacent_channels_id = np.array([
                    k%self.nb_projs_per_img for k in range(-self.nb_adj_angles, self.nb_adj_angles)
                ])

                self.channels_id = adjacent_channels_id

            # eq angles
            equiditributed_channels_id = np.array([(k * step) % self.nb_projs_per_img for k in range(1, self.input_eq_angles)])
            self.channels_id = np.concatenate((self.channels_id, equiditributed_channels_id)) if len(
                equiditributed_channels_id) > 0 else self.channels_id

            print('à'*30)
            print(self.channels_id)
            print('à' * 30)
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
                data_PVE_noisy = np.array(data['PVE_att_noisy'][channels[id],48:208,16:240],dtype=self.dtype)[invid]
                data_PVfree = np.array(data['PVfree_att'][proj_i:proj_i+1,48:208,16:240],dtype=self.dtype)
            else:
            #sino
                data_PVE_noisy, data_PVfree = np.array(data['PVE_att_noisy'][:,channels[id],:], dtype=self.dtype).transpose((1,0,2))[invid], np.array(data['PVfree'][:,proj_i:proj_i+1,:], dtype=self.dtype).transpose((1,0,2))
                data_PVE_noisy = self.pad(torch.from_numpy(data_PVE_noisy[None, :, :, :]))[0, :, :, :]
                data_PVfree = self.pad(torch.from_numpy(data_PVfree[None, :, :, :]))[0, :, :, :]
            # end sino

            data_targets, data_inputs = {},{}
            data_targets['PVfree'] = data_PVfree

            if (self.double_model and not self.test):
                if not self.sino:
                    data_PVE = np.array(data['PVE_att'][channels[id],48:208,16:240],dtype=self.dtype)[invid]
                else:
                #sino
                    data_PVE = np.array(data['PVE_att'][:,channels[id],:], dtype=self.dtype).transpose((1,0,2))[invid]
                    data_PVE = self.pad(torch.from_numpy(data_PVE[None, :, :, :]))[0, :, :, :]
                # end sino

                data_targets['PVE'] = data_PVE

                if "noise" in self.list_transforms:
                    data_PVE_noisy = self.apply_noise(data_PVE)



            data_inputs['PVE_noisy'] = data_PVE_noisy

            if self.with_rec_fp:
                if not self.sino:
                    rec_fp = np.array(data['rec_fp_att'][channels[id],48:208,16:240], dtype=self.dtype)[invid]
                else:
                #sino
                    rec_fp = np.array(data['rec_fp'][:,proj_i:proj_i+1,:], dtype = self.dtype).transpose((1,0,2))
                    rec_fp = self.pad(torch.from_numpy(rec_fp[None, :, :, :]))[0, :, :, :]
                #end sino

                data_inputs['rec_fp'] = rec_fp

            if (self.with_lesion and not self.test):
                lesion_mask = np.array(data['lesion_mask_fp'][proj_i:proj_i+1,48:208,16:240], dtype=self.dtype).astype(bool)
                data_targets['lesion_mask'] = lesion_mask

            # (forward_projected) attenuation
            if self.with_att:
                data_attmap_fp = np.array(data['attmap_fp'][channels[id],48:208,16:240], dtype=self.dtype)[invid]
                data_inputs['attmap_fp'] = data_attmap_fp


        for key in data_inputs.keys():
            data_inputs[key] = torch.from_numpy(data_inputs[key])

        for key in data_targets.keys():
            data_targets[key] = torch.from_numpy(data_targets[key])

        return data_inputs, data_targets


class CircularPadSino(torch.nn.Module):
    def __init__(self, pad: int) -> None:
        super().__init__()
        self.pad = pad

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim()==3:
            return torch.cat((input[-self.pad:,:,:], input, input[:self.pad,:,:]),dim=0)
        elif input.dim()==4:
            return torch.cat((input[:,-self.pad:,:,:], input, input[:,:self.pad,:,:]),dim=1)

class ZeroPadImgs(torch.nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim()==3:
            return torch.nn.functional.pad(input, ((self.size - input.shape[2])//2 + (self.size - input.shape[2])%2, (self.size - input.shape[2])//2,
                                               (self.size - input.shape[1])//2 + (self.size - input.shape[1])%2, (self.size - input.shape[1])//2,
                                               (self.size - input.shape[0])//2 + (self.size - input.shape[0])%2, (self.size - input.shape[0])//2),
                                       mode="constant", value=0)
        elif input.dim()==4:
            return torch.nn.functional.pad(input, ((self.size - input.shape[3])//2 + (self.size - input.shape[3])%2, (self.size - input.shape[3])//2,
                                               (self.size - input.shape[2])//2 + (self.size - input.shape[2])%2, (self.size - input.shape[2])//2,
                                               (self.size - input.shape[1])//2 + (self.size - input.shape[1])%2, (self.size - input.shape[1])//2,
                                                   0, 0),
                                           mode="constant", value=0)

class ZeroUNPadImgs(torch.nn.Module):
    def __init__(self, pad_size, inital_shape):
        super().__init__()
        self.pad_size = pad_size
        self.initial_shape = inital_shape

        self.first_0,self.first_1, self.first_2 = ((self.pad_size - self.initial_shape[0]) // 2 + (self.pad_size - self.initial_shape[0]) % 2,
        (self.pad_size - self.initial_shape[1]) // 2 + (self.pad_size - self.initial_shape[1]) % 2,
        (self.pad_size - self.initial_shape[2]) // 2 + (self.pad_size - self.initial_shape[2]) % 2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim()==3:
            return input[self.first_0:self.initial_shape[0]+self.first_0,
                   self.first_1:self.initial_shape[1]+self.first_1,
                   self.first_2:self.initial_shape[2]+self.first_2]


class SinoToSinoDataset(BaseDataset):
    def __init__(self, params, paths, filetype=None, merged=None, test=False):
        super().__init__(params, paths, filetype, merged, test)
        if params['pad']=="zero":
            self.pad = torch.nn.ConstantPad2d((0, 0, 4, 4), 0) if self.sino else torch.nn.ConstantPad2d((0, 0, 0, 0, 4, 4), 0)
        elif params['pad']=='circular':
            self.pad = CircularPadSino(4)
        else:
            self.pad = torch.nn.Identity()

        if self.params['dim'] == "2d":
            self.dim=2
        elif self.params['dim']=="3d":
            self.dim=3

        if "finetuning" in params and params['finetuning']:
            self.key_PVE_noisy = "gagarf_SC"
        else:
            self.key_PVE_noisy = "PVE_att_noisy"

        self.patches=params['patches']
        self.init_h5()

    def init_h5(self):
        self.datasetfn = self.dataset_path[0]
        self.dataseth5 = h5py.File(self.datasetfn, 'r')
        self.keys = np.array(sorted(list(self.dataseth5.keys()))).astype(np.string_)

        first_data = np.array(self.dataseth5[self.keys[0]][self.key_PVE_noisy],dtype=self.dtype)
        self.nb_projs_per_img, self.nb_pix_x, self.nb_pix_y = first_data.shape[0], first_data.shape[1], \
                                                              first_data.shape[2]

        if ((self.nb_pix_x==256) and (self.nb_pix_y==256)):
            self.fovi1,self.fovi2 = 48,208
            self.fovj1,self.fovj2 = 16,240
        elif ((self.nb_pix_x==128) and (self.nb_pix_y==128)):
            self.fovi1,self.fovi2 = 24,104
            self.fovj1,self.fovj2 = 8,120
        else:
            print(f"ERROR : invalid number of pixel. Expected nb of pixel in detector to be either (128x128) or (256x256) but found ({self.nb_pix_x}x{self.nb_pix_y})")
            exit(0)

        print(f"fov pixels : {[[self.fovi1,self.fovi2], [self.fovj1,self.fovj2]]}")

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

        if self.dim==2:
            self.channels_id = torch.Tensor([k for k in range(self.nb_projs_per_img)]).to(int)

        self.len_dataset = self.nb_src if (self.dim==3 or self.test) else self.nb_src * self.nb_projs_per_img
        # self.len_dataset = self.nb_src

        if self.patches:
            self.len_dataset=self.len_dataset * self.tile_shape[0]*self.tile_shape[1]*self.tile_shape[2]

    def get_item_h5_full_sino(self, item):
        if self.patches:
            src_i=item%self.nb_src
            i,j,k=item%self.tile_shape[0],item%self.tile_shape[1],item%self.tile_shape[2]
        else:
            if (self.dim==3 or self.test):
                src_i=item
            elif self.dim==2:
                src_i = item%self.nb_src
                proj_i = item%self.nb_projs_per_img

        with h5py.File(self.datasetfn, 'r') as f:
            data = f[self.keys[src_i]]

            data_inputs, data_targets={}, {}

            data_targets['PVfree'] = np.array(data['PVfree_att'][:,:,:],dtype=self.dtype)

            if (self.double_model and not self.test):
                data_PVE = np.array(data['PVE_att'][:,:,:], dtype=self.dtype)
                data_inputs['PVE_noisy'] = self.apply_noise(data_PVE) if 'noise' in self.list_transforms else np.array(data[self.key_PVE_noisy][:,:,:], dtype=self.dtype)
                data_targets['PVE'] = data_PVE
            else:
                data_inputs['PVE_noisy'] = np.array(data[self.key_PVE_noisy][:,:,:], dtype=self.dtype)


            if self.with_rec_fp:
                data_inputs['rec_fp'] = np.array(data['rec_fp_att'][:,:,:], dtype=self.dtype) # (120,256,256)

            if (self.with_lesion and not self.test):
                data_targets['lesion_mask']=np.array(data['lesion_mask_fp'][:,:,:], dtype=self.dtype).astype(bool)


            if self.with_att:
                # (forward_projected) attenuation
                data_inputs['attmap_fp'] = np.array(data['attmap_fp'][:,:,:], dtype=self.dtype)


        if (self.dim==2 and not self.test):
            for key in data_inputs.keys():
                data_inputs[key] = data_inputs[key][(self.channels_id+proj_i)%self.nb_projs_per_img,:,:]

            for key in data_targets.keys():
                data_targets[key] = data_targets[key][(self.channels_id+proj_i)%self.nb_projs_per_img,:,:]

        if "rot" in self.list_transforms:
            random_proj_index=np.random.randint(self.nb_projs_per_img)
            for key_inputs in data_inputs.keys():
                data_inputs[key_inputs] = np.roll(data_inputs[key_inputs], -random_proj_index,axis=0)
            for key_targets in data_targets.keys():
                data_targets[key_targets] = np.roll(data_targets[key_targets], -random_proj_index,axis=0)

        # for key_inputs in data_inputs.keys():
        #     data_inputs[key_inputs] = self.pad(torch.from_numpy(data_inputs[key_inputs]))
        # for key_targets in data_targets.keys():
        #     data_targets[key_targets] = self.pad(torch.from_numpy(data_targets[key_targets]))

        for key_inputs in data_inputs.keys():
            data_inputs[key_inputs] = data_inputs[key_inputs][:,self.fovi1:self.fovi2,self.fovj1:self.fovj2]
        for key_targets in data_targets.keys():
            data_targets[key_targets] = data_targets[key_targets][:,self.fovi1:self.fovi2,self.fovj1:self.fovj2]


        return data_inputs,data_targets

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, item):
        return self.get_item_h5_full_sino(item)

class ImgToImgDataset(BaseDataset):
    def __init__(self, params, paths, filetype=None, merged=None, test=False):
        super().__init__(params, paths, filetype, merged, test)

        self.dim=3

        if params['pad']=="zero":
            self.pad = ZeroPadImgs(128)
        else:
            self.pad = torch.nn.Identity()

        self.init_h5()

        if "vol" in self.list_transforms:
            self.get_augmentation = lambda img_size: Compose([
                    Rotate((-5, 5), (0, 0), (0, 0), p=0.5),
                    Rotate((0, 0), (-5, 5), (0, 0), p=0.5),
                    Rotate((0, 0), (0, 0), (-5, 5), p=0.5),
                    RandomCropFromBorders(crop_value=0.1, p=0.3),
                    Resize(img_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
                    Flip(0, p=0.5),
                    Flip(1, p=0.5),
                    Flip(2, p=0.5),
                    RandomRotate90((1, 2), p=0.5),
                ], p=1.0)

    def init_h5(self):
        self.datasetfn = self.dataset_path[0]
        self.dataseth5 = h5py.File(self.datasetfn, 'r')
        self.keys = sorted(list(self.dataseth5.keys()))

        first_data = np.array(self.dataseth5[self.keys[0]]['rec'],dtype=self.dtype)
        self.nb_pix_x, self.nb_pix_y, self.nb_pix_z = first_data.shape[0], first_data.shape[1], \
                                                              first_data.shape[2]

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

    def get_item_h5_img_to_img(self, item):
        src_i=item
        with h5py.File(self.datasetfn, 'r') as f:
            data = f[self.keys[src_i]]

            data_inputs, data_targets={}, {}

            data_inputs['rec'] = np.array(data['rec'],dtype=self.dtype)
            data_inputs['attmap_4mm'] = np.array(data['attmap_4mm'], dtype=self.dtype)
            data_targets['src_4mm'] = np.array(data['src_4mm'], dtype=self.dtype)


        if "vol" in self.list_transforms:
            augmentation = self.get_augmentation(data_inputs['rec'].shape)
            data = {'image': data_inputs['rec'], "image2": data_inputs['attmap_4mm'],
                    "image3": data_targets['src_4mm']}
            aug_data = augmentation(**data)
            data_inputs['rec'] = aug_data['image']
            data_inputs['attmap_4mm'] = aug_data["image2"]
            data_targets['src_4mm'] = aug_data["image3"]

        for key_inputs in data_inputs.keys():
            data_inputs[key_inputs] = self.pad(torch.from_numpy(data_inputs[key_inputs]))
        for key_targets in data_targets.keys():
            data_targets[key_targets] = self.pad(torch.from_numpy(data_targets[key_targets]))

        return data_inputs,data_targets

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, item):
        return self.get_item_h5_img_to_img(item)


def get_dataset(params, paths,filetype=None,merged=None,test=False):
    if params['inputs']=="full_sino":
        return SinoToSinoDataset(params=params,paths=paths,filetype=filetype,merged=merged,test=test)
    elif params['inputs']=="projs":
        return ProjToProjDataset(params=params,paths=paths,filetype=filetype,merged=merged,test=test)
    elif params['inputs']=="imgs":
        return ImgToImgDataset(params=params, paths=paths, filetype=filetype, merged=merged, test=test)

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
                                  num_workers=params['num_workers'],
                                  pin_memory=True,
                                  sampler=test_sampler)

    if ("validation_dataset_path" in params):
        params['finetuning']=False # FIXME tmp because no gagarf_SC data for validation dataset yet
        validation_dataset = get_dataset(params=params, paths=params['validation_dataset_path'], test=True)
        val_batchsize = params['test_batchsize']

        if params['jean_zay']:
            import idr_torch
            val_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset,
                                                                           num_replicas=idr_torch.size,
                                                                           rank=idr_torch.rank,
                                                                           shuffle=False)
        else:
            val_sampler = None

        validation_dataloader = DataLoader(dataset=validation_dataset,
                                     batch_size=val_batchsize,
                                     shuffle=False,
                                     num_workers=params['num_workers'],
                                     pin_memory=True,
                                     sampler=val_sampler)

    else:
        validation_dataloader = None

    nb_training_data = len(train_dataloader.dataset)
    nb_testing_data = len(test_dataloader.dataset)
    if params['verbose']>0:
        print(f'Number of training data : {nb_training_data}')
        print(f'Number of testing data : {nb_testing_data}')
        if validation_dataloader is not None:
            print(f'Number of validation data : {len(validation_dataset)}')
    params['nb_training_data'] = nb_training_data
    params['nb_testing_data'] = nb_testing_data
    params['nb_gpu'] = number_gpu
    params['training_mini_batchsize'] = training_batchsize
    params['test_mini_batchsize'] = test_batchsize
    params['norm'] = 'none'

    return train_dataloader, test_dataloader,validation_dataloader,params
