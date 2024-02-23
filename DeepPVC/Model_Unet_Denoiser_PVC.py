import os.path
import time

import matplotlib.pyplot as plt
import torch
from torch import optim

from . import networks, losses, helpers_data_parallelism, networks_diff, plots,networks_attention
from torch.cuda.amp import autocast, GradScaler

from .Model_base import ModelBase


class UNet_Denoiser_PVC(ModelBase):
    def __init__(self, params, from_pth=None, resume_training=False, device=None):
        assert (params['network'] == 'unet_denoiser_pvc')
        super().__init__(params, resume_training, device=device)
        self.network_type = 'unet_denoiser_pvc'
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

        self.attention = False if 'attention' not in params else params['attention']

        self.denoise = params['denoise'] if "denoise" in params else True

        self.paths= params['paths'] if "paths" in params else False

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

            self.unet_denoiser_losses,self.unet_pvc_losses = [],[]
            self.test_error = []

            self.current_epoch = 1
            self.start_epoch = 1

        self.current_iteration = 0
            # self.mean_unet_denoiser_loss, self.mean_unet_pvc_loss = torch.tensor([0], device=self.device, dtype=torch.float64), torch.tensor([0],  device=self.device,  dtype=torch.float64)
        self.mean_unet_denoiser_loss, self.mean_unet_pvc_loss = 0,0

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
            self.UNet_denoiser = networks_attention.R2AttU_Net(img_ch=self.input_channels,output_ch=self.input_channels,t=2).to(device=self.device)
            self.UNet_pvc = networks_attention.R2AttU_Net(img_ch=self.input_channels,output_ch=1,t=2).to(device=self.device)
        elif self.DCNN:
            # self.UNet_denoiser = networks.ResCNN(in_channels=self.input_channels,out_channels=self.input_channels,ngc=self.hidden_channels_unet).to(device=self.device)
            # self.UNet_pvc = networks.ResCNN(in_channels=self.input_channels, out_channels=1,ngc=self.hidden_channels_unet).to(device=self.device)
            self.UNet_denoiser = networks.vanillaCNN(input_channel=self.input_channels, ngc=self.hidden_channels_unet,
                                    dim=self.dim,init_feature_kernel=self.init_feature_kernel,
                                      nb_ed_layers=self.nb_ed_layers,
                                      output_channel=self.output_channels_denoiser, generator_activation=self.unet_activation,
                                      use_dropout=self.use_dropout, leaky_relu=self.leaky_relu,
                                      norm=self.layer_norm, residual_layer=self.residual_layer
                                      ).to(device=self.device)
            self.UNet_pvc = networks.vanillaCNN(input_channel=self.input_channels, ngc=self.hidden_channels_unet,
                                      dim=self.dim,init_feature_kernel=self.init_feature_kernel,
                                      nb_ed_layers=self.nb_ed_layers,
                                      output_channel=self.output_channels, generator_activation=self.unet_activation,
                                      use_dropout=self.use_dropout, leaky_relu=self.leaky_relu,
                                      norm=self.layer_norm, residual_layer=self.residual_layer
                                      ).to(device=self.device)
        else:
            self.UNet_denoiser = networks.UNet(input_channel=self.input_channels, ngc=self.hidden_channels_unet,paths=self.paths,
                                    dim=self.dim,init_feature_kernel=self.init_feature_kernel,
                                      nb_ed_layers=self.nb_ed_layers,
                                      output_channel=self.output_channels_denoiser, generator_activation=self.unet_activation,
                                      use_dropout=self.use_dropout, leaky_relu=self.leaky_relu,
                                      norm=self.layer_norm, residual_layer=self.residual_layer, blocks=self.ed_blocks,
                                      ResUnet=self.ResUnet,
                                               final_2dconv=False).to(device=self.device)
            self.UNet_pvc = networks.UNet(input_channel=self.input_channels, ngc=self.hidden_channels_unet,paths=self.paths,
                                      dim=self.dim,init_feature_kernel=self.init_feature_kernel,
                                      nb_ed_layers=self.nb_ed_layers,
                                      output_channel=self.output_channels, generator_activation=self.unet_activation,
                                      use_dropout=self.use_dropout, leaky_relu=self.leaky_relu,
                                      norm=self.layer_norm, residual_layer=self.residual_layer, blocks=self.ed_blocks,
                                      ResUnet=self.ResUnet,
                                      final_2dconv=self.final_2dconv, final_2dchannels=2*self.params['nb_adj_angles'] if self.final_2dconv else 0).to(device=self.device)

        if self.params['jean_zay']:
            helpers_data_parallelism.init_data_parallelism(model=self)

        if (('compile' in self.params) and (self.params['compile'] == True)):
            self.compile = True
        else:
            self.compile = False

        if (self.compile and self.for_training):
            self.UNet_denoiser = torch.compile(self.UNet_denoiser)
            self.UNet_pvc = torch.compile(self.UNet_pvc)

    def init_optimization(self):
        if self.optimizer == 'Adam':
            # self.unet_denoiser_optimizer = optim.Adam(self.UNet_denoiser.parameters(), lr=self.learning_rate)
            # self.unet_pvc_optimizer = optim.Adam(self.UNet_pvc.parameters(), lr=self.learning_rate)
            self.double_optimizer = optim.Adam(list(self.UNet_denoiser.parameters())+list(self.UNet_pvc.parameters()), lr=self.learning_rate)
        elif self.optimizer =="AdamW":
            # self.unet_denoiser_optimizer = optim.AdamW(self.UNet_denoiser.parameters(), lr=self.learning_rate)
            # self.unet_pvc_optimizer = optim.AdamW(self.UNet_pvc.parameters(), lr=self.learning_rate)
            self.double_optimizer = optim.AdamW(list(self.UNet_denoiser.parameters())+list(self.UNet_pvc.parameters()), lr=self.learning_rate)
        elif self.optimizer =="SGD":
            # self.unet_denoiser_optimizer = optim.SGD(self.UNet_denoiser.parameters(), lr=self.learning_rate,momentum=0.9)
            # self.unet_pvc_optimizer = optim.SGD(self.UNet_pvc.parameters(), lr=self.learning_rate,momentum=0.9)
            self.double_optimizer = optim.SGD(list(self.UNet_denoiser.parameters())+list(self.UNet_pvc.parameters()), lr=self.learning_rate,momentum=0.9)
        elif self.optimizer=="RMSprop":
            # self.unet_denoiser_optimizer = optim.RMSprop(self.UNet_denoiser.parameters(), lr=self.learning_rate,momentum=0.9)
            # self.unet_pvc_optimizer = optim.RMSprop(self.UNet_pvc.parameters(), lr=self.learning_rate,momentum=0.9)
            self.double_optimizer = optim.RMSprop(list(self.UNet_denoiser.parameters())+list(self.UNet_pvc.parameters()), lr=self.learning_rate,momentum=0.9)
        else:
            raise ValueError("Unknown optimizer. Choose between : Adam, AdamW, SGD, RMSprop")

        self.learning_rate_policy_infos = self.params['lr_policy']
        if self.learning_rate_policy_infos[0] == 'multiplicative':
            mult_rate = self.learning_rate_policy_infos[1]
            self.update_lr_every = self.learning_rate_policy_infos[2]
            lbda = lambda epoch: mult_rate

            # self.scheduler_unet_denoiser = optim.lr_scheduler.MultiplicativeLR(self.unet_denoiser_optimizer, lbda)
            # self.scheduler_unet_pvc = optim.lr_scheduler.MultiplicativeLR(self.unet_pvc_optimizer, lbda)
            self.scheduler = optim.lr_scheduler.MultiplicativeLR(self.double_optimizer, lbda)

    def init_losses(self):
        self.losses_params = {'recon_loss': self.params['recon_loss'],
                              'lambda_recon': self.params['lambda_recon'], 'device': self.device}
        self.losses_denoiser = losses.UNetLosses(self.losses_params)
        self.losses_pvc = losses.UNetLosses(self.losses_params)

    def input_data(self, batch_inputs, batch_targets):
        self.truePVE_noisy = batch_inputs['PVE_noisy']
        self.truePVE = batch_targets['PVE']
        self.truePVfree = batch_targets['PVfree']

        if self.with_rec_fp:
            self.true_rec_fp = batch_inputs['rec_fp']
        if self.with_att:
            self.attmap_fp = batch_inputs['attmap_fp']
        if self.with_lesion:
            self.lesion_mask_fp = batch_targets['lesion_mask']
        else:
            self.lesion_mask_fp = None

    def forward_unet_denoiser(self):
        if self.dim==2:
            input_denoiser = torch.concat((self.truePVE_noisy,self.true_rec_fp),dim=1) if self.with_rec_fp else self.truePVE_noisy
        elif self.dim==3:
            if self.with_att:
                input_denoiser = torch.concat((self.truePVE_noisy[:,None,:,:,:], self.true_rec_fp[:,None,:,:,:], self.attmap_fp[:,None,:,:,:]), dim=1) if self.with_rec_fp else torch.concat((self.truePVE_noisy[:,None,:,:,:], self.attmap_fp[:,None,:,:,:]), dim=1)
            else:
                input_denoiser = torch.concat((self.truePVE_noisy[:,None,:,:,:], self.true_rec_fp[:,None,:,:,:]), dim=1) if self.with_rec_fp else self.truePVE_noisy[:,None,:,:,:]

        self.fakePVE = self.UNet_denoiser(input_denoiser)

    def forward_unet_pvc(self):
        if self.dim==2:
            input_pvc = torch.concat((self.fakePVE,self.true_rec_fp),dim=1) if self.with_rec_fp else self.fakePVE
        elif self.dim==3:
            if self.with_att:
                input_pvc = torch.concat((self.fakePVE,self.true_rec_fp[:,None,:,:,:], self.attmap_fp[:,None,:,:,:]), dim=1) if self.with_rec_fp else torch.concat((self.fakePVE, self.attmap_fp[:,None,:,:,:]), dim=1)
            else:
                input_pvc = torch.concat((self.fakePVE,self.true_rec_fp[:,None,:,:,:]), dim=1) if self.with_rec_fp else self.fakePVE

        # self.fakePVfree = self.UNet_pvc(input_pvc.detach())
        self.fakePVfree = self.UNet_pvc(input_pvc)

    def losses_unet_denoiser(self):
        self.unet_denoiser_loss = self.losses_denoiser.get_unet_loss(target=self.truePVE, output=self.fakePVE if self.dim==2 else self.fakePVE[:,0,:,:,:],lesion_mask=self.lesion_mask_fp)

    def losses_unet_pvc(self):
        self.unet_pvc_loss = self.losses_pvc.get_unet_loss(target=self.truePVfree,output=self.fakePVfree if self.dim==2 else self.fakePVfree[:,0,:,:,:], lesion_mask=self.lesion_mask_fp)

    def backward_unet_denoiser(self):
        if self.amp:
            self.scaler.scale(self.unet_denoiser_loss).backward(retain_graph=True)
            self.scaler.step(self.unet_denoiser_optimizer)
        else:
            self.unet_denoiser_loss.backward(retain_graph=True)
            self.unet_denoiser_optimizer.step()

    def backward_unet_pvc(self):
        if self.amp:
            self.scaler.scale(self.unet_pvc_loss).backward()
            self.scaler.step(self.unet_pvc_optimizer)
        else:
            self.unet_pvc_loss.backward()
            self.unet_pvc_optimizer.step()

    def backward_double_unet(self):
        # double_loss = self.unet_denoiser_loss + self.unet_pvc_loss
        if self.amp:
            if self.denoise:
                self.scaler.scale(self.unet_denoiser_loss + self.unet_pvc_loss).backward()
            else:
                self.scaler.scale(self.unet_pvc_loss).backward()
            # if (self.current_iteration%3==0):
            self.scaler.step(self.double_optimizer)
            self.scaler.update()
            self.double_optimizer.zero_grad(set_to_none=True)
        else:
            (self.unet_denoiser_loss + self.unet_pvc_loss).backward()
            self.double_optimizer.step()
            self.double_optimizer.zero_grad(set_to_none=True)

    def forward(self, batch):
        if self.with_rec_fp:
            if self.with_att:
                truePVEnoisy,true_rec_fp,attmap_fp = batch['PVE_noisy'], batch['rec_fp'], batch['attmap_fp']
            else:
                truePVEnoisy, true_rec_fp = batch['PVE_noisy'], batch['rec_fp']

            # ----------------------------

            # patch_size=(32,64,64)
            # truePVEnoisy = truePVEnoisy.unfold(1, patch_size[0], patch_size[0]).unfold(2, patch_size[1], patch_size[1]).unfold(3, patch_size[2], patch_size[2])
            # true_rec_fp = true_rec_fp.unfold(1, patch_size[0], patch_size[0]).unfold(2, patch_size[1], patch_size[1]).unfold(3, patch_size[2], patch_size[2])
            # unfold_shape=truePVEnoisy.size()
            # truePVEnoisy=truePVEnoisy.contiguous().view(-1, patch_size[0], patch_size[1], patch_size[2])
            # true_rec_fp=true_rec_fp.contiguous().view(-1, patch_size[0], patch_size[1], patch_size[2])

            if self.dim==2:
                truePVEnoisy = torch.concat((truePVEnoisy, true_rec_fp), dim=1)
            elif self.dim==3:
                if self.with_att:
                    truePVEnoisy = torch.concat((truePVEnoisy[:,None,:,:,:], true_rec_fp[:,None,:,:,:], attmap_fp[:,None,:,:,:]),dim=1)
                else:
                    truePVEnoisy = torch.concat((truePVEnoisy[:,None,:,:,:], true_rec_fp[:,None,:,:,:]),dim=1)
        else:
            if self.with_att:
                truePVEnoisy,attmap_fp = batch['PVE_noisy'], batch['attmap_fp']
            else:
                truePVEnoisy = batch['PVE_noisy']

            if self.dim==2:
                truePVEnoisy = batch['PVE_noisy']
            elif self.dim==3:
                if self.with_att:
                    truePVEnoisy = torch.concat((truePVEnoisy[:,None,:,:,:], attmap_fp[:,None,:,:,:]),dim=1)
                else:
                    truePVEnoisy = truePVEnoisy[:,None,:,:,:]

        with autocast(enabled=self.amp,dtype=torch.float16):
            fakePVE = self.UNet_denoiser(truePVEnoisy)

        if self.with_rec_fp:
            if self.dim==2:
                fakePVE = torch.concat((fakePVE, true_rec_fp), dim=1)
            elif self.dim==3:
                if self.with_att:
                    fakePVE = torch.concat((fakePVE, true_rec_fp[:,None,:,:,:], attmap_fp[:,None,:,:,:]), dim=1)
                else:
                    fakePVE = torch.concat((fakePVE, true_rec_fp[:,None,:,:,:]), dim=1)
        else:
            if self.dim == 3:
                if self.with_att:
                    fakePVE = torch.concat((fakePVE,attmap_fp[:, None, :, :, :]), dim=1)


        if self.dim==2:
            with autocast(enabled=self.amp,dtype=torch.float16):
                output = self.UNet_pvc(fakePVE)
            return output
        elif self.dim==3:
            with autocast(enabled=self.amp,dtype=torch.float16):
                output= self.UNet_pvc(fakePVE)[:,0,:,:,:]
            # output=output.view(unfold_shape)
            # output_c,output_h,output_w=unfold_shape[1]*unfold_shape[4],unfold_shape[2]*unfold_shape[5],unfold_shape[3]*unfold_shape[6]
            # output=output.permute(0,1,4,2,5,3,6).contiguous()
            # output = output.view(1, output_c, output_h, output_w)
            return output



    def optimize_parameters(self):
        # Unet denoiser update
        self.set_requires_grad(self.UNet_denoiser, requires_grad=True)
        self.set_requires_grad(self.UNet_pvc, requires_grad=True)
        # self.double_optimizer.zero_grad(set_to_none=True)
        # self.unet_denoiser_optimizer.zero_grad(set_to_none=True)
        # with autocast(enabled=self.amp):
            # self.forward_unet_denoiser()
            # self.losses_unet_denoiser()
        # self.backward_unet_denoiser()

        # Unet pvc update
        # self.set_requires_grad(self.UNet_pvc, requires_grad=True)
        # self.unet_pvc_optimizer.zero_grad(set_to_none=True)
        # with autocast(enabled=self.amp):
        #     self.forward_unet_pvc()
            # self.losses_unet_pvc()
        # self.backward_unet_pvc()

        with autocast(enabled=self.amp, dtype=torch.float16):
            self.forward_unet_denoiser()
            if self.denoise:
                self.losses_unet_denoiser()
            self.forward_unet_pvc()
            self.losses_unet_pvc()
        self.backward_double_unet()

        if self.denoise:
            self.mean_unet_denoiser_loss += self.unet_denoiser_loss.item()
        self.mean_unet_pvc_loss += self.unet_pvc_loss.item()
        self.current_iteration += 1

        # self.iter_loss.append((self.unet_denoiser_loss + self.unet_pvc_loss).item())

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

        if self.with_lesion:
            del self.truePVE, self.truePVfree,self.lesion_mask_fp
        else:
            del self.truePVE, self.truePVfree
            del self.lesion_mask_fp

        del self.fakePVE, self.fakePVfree

    def update_epoch(self):
        self.unet_denoiser_losses.append(self.mean_unet_denoiser_loss / self.current_iteration)
        self.unet_pvc_losses.append(self.mean_unet_pvc_loss / self.current_iteration)
        if self.verbose > 1:
            print(f'Unet denoiser loss : {round(self.unet_denoiser_losses[-1], 5)} / pvc loss : {round(self.unet_pvc_losses[-1], 5)}')

        if self.current_epoch % self.update_lr_every == 0:
            # self.scheduler_unet_denoiser.step()
            # self.scheduler_unet_pvc.step()
            self.scheduler.step()

        if self.verbose > 1:
            # print(f'next lr : {self.scheduler_unet_denoiser.get_last_lr()}')
            print(f'next lr : {self.scheduler.get_last_lr()}')

        self.current_epoch += 1
        self.current_iteration = 0
        self.mean_unet_denoiser_loss, self.mean_unet_pvc_loss = 0,0

        # fig,ax = plt.subplots()
        # ax.plot(self.iter_loss)
        # plt.show()
        # self.iter_loss=[]

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
                    'unet_denoiser': self.UNet_denoiser.module.state_dict() if jean_zay else self.UNet_denoiser.state_dict(),
                    'unet_pvc': self.UNet_pvc.module.state_dict() if jean_zay else self.UNet_pvc.state_dict(),
                    # 'unet_denoiser_opt': self.unet_denoiser_optimizer.state_dict(),
                    # 'unet_pvc_opt': self.unet_pvc_optimizer.state_dict(),
                    'double_optimizer': self.double_optimizer.state_dict(),
                    'unet_denoiser_losses': self.unet_denoiser_losses,
                    'unet_pvc_losses': self.unet_pvc_losses,
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

        if hasattr(self.UNet_denoiser, 'module'):
            self.UNet_denoiser.module.load_state_dict(checkpoint['unet_denoiser'])
            self.UNet_pvc.module.load_state_dict(checkpoint['unet_pvc'])
        else:
            self.UNet_denoiser.load_state_dict(checkpoint['unet_denoiser'])
            self.UNet_pvc.load_state_dict(checkpoint['unet_pvc'])
        self.unet_denoiser_losses,self.unet_pvc_losses = checkpoint['unet_denoiser_losses'],checkpoint['unet_pvc_losses']
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
        self.UNet_denoiser.eval()
        self.UNet_pvc.eval()

    def switch_train(self):
        self.UNet_denoiser.train()
        self.UNet_pvc.train()

    def switch_device(self, device):
        self.device = device
        self.UNet_denoiser = self.UNet_denoiser.to(device=device)
        self.UNet_pvc = self.UNet_pvc.to(device=device)

    def show_infos(self, mse=False):
        formatted_params = self.format_params()
        print('*' * 80)
        print('PARAMETRES (json param file) : \n')
        print(formatted_params)
        print('*' * 80)

        print('MEAN SQUARE ERROR on TEST DATA')
        print(f'list of errors on test data :{self.test_error} ')
        print('*' * 20 + "DENOISER" + '*'*20)
        print(self.UNet_denoiser)
        nb_params = sum(p.numel() for p in self.UNet_denoiser.parameters())
        print(f'NUMBER OF PARAMERS : {nb_params}')

        print('*' * 20 + "PVC" + '*'*20)
        print(self.UNet_pvc)
        nb_params = sum(p.numel() for p in self.UNet_pvc.parameters())
        print(f'NUMBER OF PARAMERS : {nb_params}')
        if hasattr(self, "losses_denoiser"):
            print('Losses : ')
            print(self.losses_params)
            print('Denoiser loss : ')
            print(self.losses_denoiser)
            print('PVC loss : ')
            print(self.losses_pvc)
            print('*' * 80)


    def plot_losses(self, save, wait, title):
        plots.plot_losses_UNet(unet_losses=self.unet_denoiser_losses, test_mse=[], save=save, wait=True, title=title)
        plots.plot_losses_UNet(unet_losses=self.unet_pvc_losses, test_mse=[], save=save, wait=wait, title=title)

        fig,ax =plt.subplots()
        ax.plot(self.unet_denoiser_losses, label="denoiser")
        ax.plot(self.unet_pvc_losses, label="pvc")
        fig.legend()