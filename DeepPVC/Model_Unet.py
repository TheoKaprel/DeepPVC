import os.path
import time

import itk
import torch
from torch import optim

from . import networks, losses, helpers_data_parallelism, plots,networks_attention
from torch.cuda.amp import autocast, GradScaler

from .Model_base import ModelBase


class UNetModel(ModelBase):
    def __init__(self, params, from_pth=None, resume_training=False, device=None):
        assert (params['network'] == 'unet')
        super().__init__(params, resume_training, device=device)
        self.network_type = 'unet'
        self.verbose = params['verbose']

        if "dim" in params:
            if params["dim"]=="2d":
                self.dim=2
            elif params["dim"]=="3d":
                self.dim=3
        else:
            self.dim=2

        self.init_feature_kernel = params['init_feature_kernel']
        self.nb_ed_layers = params['nb_ed_layers']
        if "ed_blocks" in params:
            self.ed_blocks = params["ed_blocks"]
        else:
            self.ed_blocks = "conv-relu-norm"

        self.with_rec_fp = params['with_rec_fp']

        self.hidden_channels_unet = params["hidden_channels_unet"]
        self.unet_activation = params["unet_activation"]

        self.layer_norm = params['layer_norm']
        self.residual_layer = params['residual_layer']
        self.ResUnet = params['resunet']

        self.final_2dconv = True if (params['dim']=="3d" and params['inputs']=="projs") else False

        self.DCNN = params['DCNN'] if 'DCNN' in params else False

        self.img_to_img = (params['inputs'] == "imgs")

        self.paths= params['paths'] if "paths" in params else False


        self.attention = False if 'attention' not in params else params['attention']

        if from_pth:
            self.for_training=False
            self.init_model()
            if self.verbose > 1:
                print(
                    'normalement self.load_model(from_pth) mais là non, on le fait juste apres l initialisation des gpus etc')
        else:
            self.for_training = True
            self.init_model()
            self.init_optimization()
            self.init_losses()

            self.unet_losses = []
            self.test_error = []

            self.current_epoch = 1
            self.start_epoch = 1

        self.current_iteration = 0
            # self.mean_unet_denoiser_loss, self.mean_unet_pvc_loss = torch.tensor([0], device=self.device, dtype=torch.float64), torch.tensor([0],  device=self.device,  dtype=torch.float64)
        self.mean_unet_loss = 0

        # self.iter_loss=[]

        self.amp = self.params['amp']
        if self.amp:
            self.scaler = GradScaler()
        self.autocat_losses = self.amp

    def init_model(self):
        if self.verbose > 0:
            print(f'models device is supposed to be : {self.device}')
        if self.attention:
            # self.UNet = networks_diff.AttentionResUnet(init_dim=self.hidden_channels_unet, out_dim=1,
            #                                            channels=self.input_channels, dim_mults=(1, 2, 4, 8)).to(
            #     device=self.device)
            # self.UNet = networks_attention.R2AttU_Net(img_ch=self.input_channels,output_ch=self.input_channels,t=2).to(device=self.device)
            self.UNet = networks.UNet(input_channel=self.input_channels, ngc=self.hidden_channels_unet,
                                      dim=self.dim,init_feature_kernel=self.init_feature_kernel,
                                      nb_ed_layers=self.nb_ed_layers,
                                      output_channel=self.output_channels, generator_activation=self.unet_activation,
                                      use_dropout=self.use_dropout, leaky_relu=self.leaky_relu,
                                      norm=self.layer_norm, residual_layer=0 if self.residual_layer else -1, blocks=self.ed_blocks,
                                      ResUnet=self.ResUnet,AttentionUnet = True,
                                      final_2dconv=self.final_2dconv, final_2dchannels=2*self.params['nb_adj_angles'] if self.final_2dconv else 0,
                                      paths=self.paths).to(device=self.device)
        elif self.DCNN:
            # self.UNet = networks.vanillaCNN(input_channel=self.input_channels, ngc=self.hidden_channels_unet,
            #                           dim=self.dim,init_feature_kernel=self.init_feature_kernel,
            #                           nb_ed_layers=self.nb_ed_layers,
            #                           output_channel=self.output_channels, generator_activation=self.unet_activation,
            #                           use_dropout=self.use_dropout, leaky_relu=self.leaky_relu,
            #                           norm=self.layer_norm, residual_layer=self.residual_layer
            #                           ).to(device=self.device)
            self.UNet = networks.CNN(input_channel=self.input_channels, ngc=self.hidden_channels_unet,
                                      kernel=self.init_feature_kernel,
                                      nb_ed_layers=self.nb_ed_layers,
                                      output_channel=self.output_channels, generator_activation=self.unet_activation,
                                      leaky_relu=self.leaky_relu,
                                      norm=self.layer_norm, residual_layer=0 if self.residual_layer else -1,
                                      blocks=self.ed_blocks,
                                      ResUnet=self.ResUnet,
                                      paths=self.paths).to(device=self.device)


        elif False:
            self.UNet = networks.UNET_3D_2D(input_channel=self.input_channels,residual_layer=self.residual_layer,
                                            final_2dchannels=2*self.params['nb_adj_angles']).to(device=self.device)
        else:
            self.UNet = networks.UNet(input_channel=self.input_channels, ngc=self.hidden_channels_unet,
                                      dim=self.dim,init_feature_kernel=self.init_feature_kernel,
                                      nb_ed_layers=self.nb_ed_layers,
                                      output_channel=self.output_channels, generator_activation=self.unet_activation,
                                      use_dropout=self.use_dropout, leaky_relu=self.leaky_relu,
                                      norm=self.layer_norm, residual_layer=0 if self.residual_layer else -1, blocks=self.ed_blocks,
                                      ResUnet=self.ResUnet,AttentionUnet=False,
                                      final_2dconv=self.final_2dconv, final_2dchannels=2*self.params['nb_adj_angles'] if self.final_2dconv else 0,
                                      paths=self.paths).to(device=self.device)

        if "init" not in self.params:
            self.params["init"] = "none"
        networks.init_weights(net=self.UNet, init_type=self.params["init"])



        if self.params['jean_zay']:
            helpers_data_parallelism.init_data_parallelism(model=self)

        if (('compile' in self.params) and (self.params['compile'] == True)):
            self.compile = True
        else:
            self.compile = False

        if (self.compile and self.for_training):
            self.UNet = torch.compile(self.UNet)

    def init_optimization(self):
        if self.optimizer == 'Adam':
            self.double_optimizer = optim.Adam(self.UNet.parameters(), lr=self.learning_rate)
        elif self.optimizer =="AdamW":
            self.double_optimizer = optim.AdamW(self.UNet.parameters(), lr=self.learning_rate)
        elif self.optimizer =="SGD":
            self.double_optimizer = optim.SGD(self.UNet.parameters(), lr=self.learning_rate,momentum=0.9)
        elif self.optimizer=="RMSprop":
            self.double_optimizer = optim.RMSprop(self.UNet.parameters(), lr=self.learning_rate,momentum=0.9)
        else:
            raise ValueError("Unknown optimizer. Choose between : Adam, AdamW, SGD, RMSprop")

        self.learning_rate_policy_infos = self.params['lr_policy']
        if self.learning_rate_policy_infos[0] == 'multiplicative':
            mult_rate = self.learning_rate_policy_infos[1]
            self.update_lr_every = self.learning_rate_policy_infos[2]
            lbda = lambda epoch: mult_rate
            self.scheduler = optim.lr_scheduler.MultiplicativeLR(self.double_optimizer, lbda)
        elif self.learning_rate_policy_infos[0] =="reduceplateau":
            factor=self.learning_rate_policy_infos[1]
            patience=self.learning_rate_policy_infos[2]

            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.double_optimizer, 'min',
                                                                  factor=factor,patience=patience)
            self.update_lr_every=1

    def init_losses(self):

        if self.with_conv_loss:
            psf_itk = itk.imread(self.params['psf'])
            psf_torch = torch.Tensor(itk.array_from_image(psf_itk)).to(self.device)
            self.conv_psf = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=psf_torch.shape, stride=(1, 1, 1),
                                   padding=((psf_torch.shape[0] - 1) // 2, (psf_torch.shape[1] - 1) // 2,
                                            (psf_torch.shape[2] - 1) // 2), bias=False).to(self.device)
            self.conv_psf.weight.data = psf_torch[None, None, :, :, :]
        else:
            self.conv_psf = None

        self.losses_params = {'recon_loss': self.params['recon_loss'],
                              'lambda_recon': self.params['lambda_recon'], 'device': self.device}
        self.losses = losses.UNetLosses(self.losses_params)

    def input_data(self, batch_inputs, batch_targets):
        self.truePVE_noisy = batch_inputs['PVE_noisy'] if (self.img_to_img == False) else batch_inputs['rec']
        self.truePVfree = batch_targets['PVfree'] if (self.img_to_img == False) else batch_targets['src_4mm']

        if self.with_rec_fp:
            self.true_rec_fp = batch_inputs['rec_fp']
        if self.with_att:
            self.attmap_fp = batch_inputs['attmap_fp'] if (self.img_to_img == False) else batch_inputs['attmap_4mm']
        self.lesion_mask_fp = batch_targets['lesion_mask'] if self.with_lesion else None

        self.normalize_data()

    def normalize_data(self):
        if self.params['data_normalisation']=="3d_max":
            self.norm = self.truePVE_noisy.amax((1,2,3))

            self.truePVE_noisy = self.truePVE_noisy / self.norm[:,None,None,None]
            if self.with_att:
                self.attmap_fp = self.attmap_fp / self.attmap_fp.amax((1,2,3))[:,None,None,None]

        elif self.params['data_normalisation'] in ["3d_sum", "3d_softmax"]:
            self.norm = self.truePVE_noisy.sum((1,2,3))

            self.truePVE_noisy = self.truePVE_noisy / self.truePVE_noisy.amax((1,2,3))[:,None,None,None]
            if self.with_att:
                self.attmap_fp = self.attmap_fp / self.attmap_fp.amax((1,2,3))[:,None,None,None]
        else:
            self.norm = None


    def denormalize_data(self):
        if self.params['data_normalisation'] == "3d_max":
            self.fakePVfree = self.fakePVfree  * self.norm[:,None,None,None,None]
        elif self.params['data_normalisation'] == "3d_sum":
            self.fakePVfree = (self.fakePVfree / self.fakePVfree.sum((1,2,3,4))[:,None,None,None,None]) * self.norm[:,None,None,None,None]
        elif self.params['data_normalisation'] == "3d_softmax":
            self.fakePVfree = torch.exp(self.fakePVfree) / torch.exp(self.fakePVfree).sum((1,2,3,4))[:,None,None,None,None]  * self.norm[:,None,None,None,None]


    def forward_unet(self):
        if self.dim==2:
            input = torch.concat((self.truePVE_noisy,self.true_rec_fp),dim=1) if self.with_rec_fp else self.truePVE_noisy
        elif self.dim==3:
            if self.with_att:
                input = torch.concat((self.truePVE_noisy[:,None,:,:,:], self.true_rec_fp[:,None,:,:,:], self.attmap_fp[:,None,:,:,:]), dim=1) if self.with_rec_fp else torch.concat((self.truePVE_noisy[:,None,:,:,:], self.attmap_fp[:,None,:,:,:]), dim=1)
            else:
                input = torch.concat((self.truePVE_noisy[:,None,:,:,:], self.true_rec_fp[:,None,:,:,:]), dim=1) if self.with_rec_fp else self.truePVE_noisy[:,None,:,:,:]

        self.fakePVfree = self.UNet(input)
        self.denormalize_data()

    def losses_unet(self):
        self.unet_loss = self.losses.get_unet_loss(target=self.truePVfree,
                                                   output=self.fakePVfree if self.dim==2 else self.fakePVfree[:,0,:,:,:],
                                                   lesion_mask=self.lesion_mask_fp,
                                                   conv_psf=self.conv_psf,
                                                   input_rec = self.truePVE_noisy)

    def backward_unet(self):
        if self.amp:
            self.scaler.scale(self.unet_loss).backward()
            self.scaler.step(self.double_optimizer)
            self.scaler.update()
            self.double_optimizer.zero_grad(set_to_none=True)
        else:
            (self.unet_loss).backward()
            self.double_optimizer.step()
            self.double_optimizer.zero_grad(set_to_none=True)

    def forward(self, batch):
        self.truePVE_noisy = batch['PVE_noisy'] if (self.img_to_img == False) else batch['rec']

        if self.with_rec_fp:
            self.true_rec_fp = batch['rec_fp']
        if self.with_att:
            self.attmap_fp = batch['attmap_fp'] if (self.img_to_img == False) else batch['attmap_4mm']

        self.normalize_data()

        if self.with_rec_fp:
            # ----------------------------
            if self.dim==2:
                input = torch.concat((self.truePVE_noisy, self.true_rec_fp), dim=1)
            elif self.dim==3:
                if self.with_att:
                    input = torch.concat((self.truePVE_noisy[:,None,:,:,:], self.true_rec_fp[:,None,:,:,:], self.attmap_fp[:,None,:,:,:]),dim=1)
                else:
                    input = torch.concat((self.truePVE_noisy[:,None,:,:,:], self.true_rec_fp[:,None,:,:,:]),dim=1)
        else:
            if self.dim==2:
                input = batch[0]
            elif self.dim==3:
                if self.with_att:
                    input = torch.concat((self.truePVE_noisy[:,None,:,:,:], self.attmap_fp[:,None,:,:,:]),dim=1)
                else:
                    input = self.truePVE_noisy[:,None,:,:,:]

        with autocast(enabled=self.amp,dtype=torch.float16):
            self.fakePVfree = self.UNet(input)
            self.denormalize_data()

        if self.dim==2:
            return self.fakePVfree
        elif self.dim==3:
            return self.fakePVfree[:,0,:,:,:]


    def optimize_parameters(self):
        self.double_optimizer.zero_grad(set_to_none=True)

        # Unet denoiser update
        self.set_requires_grad(self.UNet, requires_grad=True)

        with autocast(enabled=self.amp, dtype=torch.float16):
            self.forward_unet()
            self.losses_unet()
        self.backward_unet()

        self.mean_unet_loss += self.unet_loss.item()
        self.current_iteration += 1
        self.del_variables()


    def del_variables(self):
        if self.with_rec_fp:
            if self.with_att:
                del self.truePVE_noisy,self.true_rec_fp,self.attmap_fp
            else:
                del self.truePVE_noisy,self.true_rec_fp
        else:
            if self.with_att:
                del self.truePVE_noisy,self.attmap_fp
            else:
                del self.truePVE_noisy


        del self.truePVfree,self.lesion_mask_fp


        del self.fakePVfree

    def update_epoch(self):
        self.unet_losses.append(self.mean_unet_loss / self.current_iteration)
        if self.verbose > 1:
            print(f'Unet loss : {round(self.unet_losses[-1], 5)}')

        if self.current_epoch % self.update_lr_every == 0:
            if self.learning_rate_policy_infos[0]=="multiplicative":
                self.scheduler.step()
            elif self.learning_rate_policy_infos[0]=="reduceplateau":
                self.scheduler.step(self.test_error[-1][1])


        if self.verbose > 1:
            # print(f'next lr : {self.scheduler.get_last_lr()}')
            print(f'next lr : {self.double_optimizer.param_groups[0]["lr"] }')

        self.current_epoch += 1
        self.current_iteration = 0
        self.mean_unet_loss = 0

    def save_model(self, output_path=None, save_json=False):
        self.params['start_epoch'] = self.start_epoch
        self.params['current_epoch'] = self.current_epoch
        jean_zay = self.params['jean_zay']

        if output_path is None:
            if self.output_folder:
                output_path = os.path.join(self.output_folder, self.output_pth)
            else:
                raise ValueError("Error: no output_folder specified")

        torch.save({'saving_date': time.asctime(),
                    'epoch': self.current_epoch,
                    'unet': self.UNet.module.state_dict() if jean_zay else self.UNet.state_dict(),
                    'double_optimizer': self.double_optimizer.state_dict(),
                    'unet_losses': self.unet_losses,
                    'test_error': self.test_error,
                    'val_mse':self.val_error_MSE,
                    'val_mae': self.val_error_MAE,
                    'params': self.params
                    }, output_path)
        if self.verbose > 0:
            print(f'Model saved at : {output_path}')

        if save_json:
            output_json = output_path[:-4] + '.json'
            formatted_params = self.format_params()
            jsonFile = open(output_json, "w")
            jsonFile.write(formatted_params)
            jsonFile.close()

    def load_model(self, pth_path, new_lr=None):
        if self.verbose > 0:
            print(f'Loading Model from {pth_path}... ')
        checkpoint = torch.load(pth_path, map_location=self.device)

        if hasattr(self.UNet, 'module'):
            self.UNet.module.load_state_dict(checkpoint['unet'])
        else:
            self.UNet.load_state_dict(checkpoint['unet'])
        self.unet_losses = checkpoint['unet_losses']
        self.test_error = checkpoint['test_error']

        self.val_error_MSE = checkpoint['val_mse'] if 'val_mse' in checkpoint else []
        self.val_error_MAE = checkpoint['val_mae'] if 'val_mae' in checkpoint else []
        self.current_epoch = checkpoint['epoch']

        if self.resume_training:
            if (new_lr is not None):
                print(f"NEW LEARNING RATE FOR RESUME TRAINING : {self.learning_rate}")
            else:
                self.learning_rate = checkpoint['double_optimizer']['param_groups'][0]['lr']

            self.init_optimization()
            self.init_losses()
            self.double_optimizer.load_state_dict(checkpoint['double_optimizer'])
            for g in self.double_optimizer.param_groups:
                g['lr'] = self.learning_rate
            self.start_epoch = self.current_epoch

    def switch_eval(self):
        self.UNet.eval()

    def switch_train(self):
        self.UNet.train()


    def switch_device(self, device):
        self.device = device
        self.UNet= self.UNet.to(device=device)

    def show_infos(self, mse=False):
        formatted_params = self.format_params()
        print('*' * 80)
        print('PARAMETRES (json param file) : \n')
        print(formatted_params)
        print('*' * 80)

        print('MEAN SQUARE ERROR on TEST DATA')
        print(f'list of errors on test data :{self.test_error} ')
        print('*' * 20 + "UNET" + '*'*20)
        print(self.UNet)
        nb_params = sum(p.numel() for p in self.UNet.parameters())
        print(f'NUMBER OF PARAMERS : {nb_params}')

        if hasattr(self, "losses"):
            print('Losses : ')
            print(self.losses_params)
            print('loss : ')
            print(self.losses)
            print('*' * 80)


    def plot_losses(self, save, wait, title):
        plots.plot_losses_UNet(unet_losses=self.unet_losses, test_mse=[], save=save, wait=wait, title=title)