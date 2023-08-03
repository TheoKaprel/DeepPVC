#!/usr/bin/env python3


import os.path
import time
import torch
from torch import optim

from . import losses, helpers_data_parallelism, networks_diff, plots,functions_diffusion
from torch.cuda.amp import autocast, GradScaler

from .Model_base import ModelBase


class Diffusion_UNet(ModelBase):
    def __init__(self, params, from_pth=None, resume_training=False, device=None):
        assert (params['network'] == 'diffusion')
        super().__init__(params, resume_training, device=device)
        self.network_type = 'diffusion'
        self.verbose = params['verbose']
        self.init_feature_kernel = params['init_feature_kernel']
        self.nb_ed_layers = params['nb_ed_layers']
        self.dim_mults = [2**k for k in range(self.nb_ed_layers)]
        self.hidden_channels_unet = params["hidden_channels_unet"]

        self.timesteps = params['timesteps']
        functions_diffusion.init_alphas_betas_t(timesteps_=self.timesteps)

        self.init_model()

        if from_pth:
            if self.verbose > 1:
                print(
                    'normalement self.load_model(from_pth) mais là non, on le fait juste apres l initialisation des gpus etc')
        else:
            self.init_optimization()

            self.diffusion_losses = []
            self.test_error = []

            self.current_epoch = 1
            self.start_epoch = 1

        self.current_iteration = 0
        self.mean_diffusion_loss = 0

        self.amp = self.params['amp']
        if self.amp:
            self.scaler = GradScaler()
        self.autocat_losses = self.amp

    def init_model(self):
        if self.verbose > 0:
            print(f'models device is supposed to be : {self.device}')
        self.Diffusion_Unet = networks_diff.Diffusion_Unet(init_dim=self.hidden_channels_unet,dim = self.hidden_channels_unet,
                                                             out_dim=1,dim_mults=self.dim_mults,self_condition=True,
                                                             input_channels=self.input_channels,
                                                             resnet_block_groups=2).to(device=self.device)

        # self.Diffusion_Unet = networks_diff.SimpleUnet(init_dim=self.hidden_channels_unet,
        #                                                       out_dim=1,dim_mults=self.dim_mults,
        #                                                       input_channels=self.input_channels).to(device=self.device)

        if self.params['jean_zay']:
            helpers_data_parallelism.init_data_parallelism(model=self)

    def init_optimization(self):
        if self.optimizer == 'Adam':
            self.diffusion_optimizer = optim.Adam(self.Diffusion_Unet.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Unknown optimizer. Choose between : Adam")

        self.learning_rate_policy_infos = self.params['lr_policy']
        if self.learning_rate_policy_infos[0] == 'multiplicative':
            mult_rate = self.learning_rate_policy_infos[1]
            self.update_lr_every = self.learning_rate_policy_infos[2]
            lbda = lambda epoch: mult_rate

            self.scheduler_diffusion = optim.lr_scheduler.MultiplicativeLR(self.diffusion_optimizer, lbda)

    def input_data(self, batch_inputs, batch_targets):
        self.truePVE_noisy = batch_inputs
        self.truePVfree = batch_targets
        self.batch_size = self.truePVE_noisy.shape[0]

    def forward(self, batch):
        if batch.dim() == 4:
            truePVEnoisy = batch
        elif batch.dim() == 5:
            truePVEnoisy = batch[:,0,:,:,:]

        output = torch.zeros((truePVEnoisy.shape[0], 1, truePVEnoisy.shape[2], truePVEnoisy.shape[3]))
        for i in range(truePVEnoisy.shape[0]):
            samples_i = functions_diffusion.sample(model=self.Diffusion_Unet,cond = truePVEnoisy[i:i+1,:,:,:],shape = [1,1,truePVEnoisy.shape[2], truePVEnoisy.shape[3]])
            output[i:i+1,:,:,:] = samples_i[-1]
        return output


    def optimize_parameters(self):
        self.set_requires_grad(self.Diffusion_Unet, requires_grad=True)
        self.diffusion_optimizer.zero_grad(set_to_none=True)

        t = torch.randint(0, self.timesteps, (self.batch_size,), device=self.device).long()

        # todo : AMP,grad_scaler
        loss = functions_diffusion.p_losses(denoise_model=self.Diffusion_Unet,
                        x_start=self.truePVfree,t=t,
                        loss_type="l1",cond = self.truePVE_noisy)

        loss.backward()
        self.diffusion_optimizer.step()

        self.mean_diffusion_loss += loss.item()
        self.current_iteration+=1



    def update_epoch(self):
        self.diffusion_losses.append(self.mean_diffusion_loss / self.current_iteration)
        if self.verbose > 1:
            print(f'Diffusion loss : {round(self.diffusion_losses[-1], 5)}')

        if self.current_epoch % self.update_lr_every == 0:
            self.scheduler_diffusion.step()

        if self.verbose > 1:
            print(f'next lr : {self.scheduler_diffusion.get_last_lr()}')

        self.current_epoch += 1
        self.current_iteration = 0
        self.mean_diffusion_loss = 0

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
                    'diffusion_unet': self.Diffusion_Unet.module.state_dict() if jean_zay else self.Diffusion_Unet.state_dict(),
                    'diffusion_opt': self.diffusion_optimizer.state_dict(),
                    'diffusion_losses': self.diffusion_losses,
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

    def load_model(self, pth_path):
        if self.verbose > 0:
            print(f'Loading Model from {pth_path}... ')
        checkpoint = torch.load(pth_path, map_location=self.device)

        if hasattr(self.Diffusion_Unet, 'module'):
            self.Diffusion_Unet.module.load_state_dict(checkpoint['diffusion_unet'])
        else:
            self.Diffusion_Unet.load_state_dict(checkpoint['diffusion_unet'])
        self.diffusion_losses = checkpoint['diffusion_losses']
        self.test_error = checkpoint['test_error']

        self.val_error_MSE = checkpoint['val_mse'] if 'val_mse' in checkpoint else []
        self.val_error_MAE = checkpoint['val_mae'] if 'val_mae' in checkpoint else []
        self.current_epoch = checkpoint['epoch']

        if self.resume_training:
            self.learning_rate = checkpoint['diffusion_opt']['param_groups'][0]['lr']

            self.init_optimization()
            self.init_losses()
            self.diffusion_optimizer.load_state_dict(checkpoint['diffusion_opt'])
            for g in self.diffusion_optimizer.param_groups:
                g['lr'] = self.scheduler_diffusion.get_last_lr()[0]
            self.start_epoch = self.current_epoch

    def switch_eval(self):
        self.Diffusion_Unet.eval()

    def switch_train(self):
        self.Diffusion_Unet.train()

    def switch_device(self, device):
        self.device = device
        self.Diffusion_Unet = self.Diffusion_Unet.to(device=device)

    def show_infos(self, mse=False):
        formatted_params = self.format_params()
        print('*' * 80)
        print('PARAMETRES (json param file) : \n')
        print(formatted_params)
        print('*' * 80)

        print('MEAN SQUARE ERROR on TEST DATA')
        print(f'list of errors on test data :{self.test_error} ')
        print('*' * 20 + "DIFFUSION UNET" + '*'*20)
        print(self.Diffusion_Unet)
        nb_params = sum(p.numel() for p in self.Diffusion_Unet.parameters())
        print(f'NUMBER OF PARAMERS : {nb_params}')
        if hasattr(self, "losses_diffusion"):
            print('Losses : ')
            print(self.losses_params)
            print('Denoiser loss : ')
            print(self.losses_diffusion)
            print('*' * 80)


    def plot_losses(self, save, wait, title):
        plots.plot_losses_UNet(unet_losses=self.diffusion_losses, test_mse=[], save=save, wait=wait, title=title)
