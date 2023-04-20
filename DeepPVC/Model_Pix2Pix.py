#!/usr/bin/env python3

from .Model_base import ModelBase

import os.path
import time
import torch
from torch import optim

from . import networks, losses,plots,helpers_data_parallelism, networks_diff
from torch.cuda.amp import autocast, GradScaler


class Pix2PixModel(ModelBase):
    def __init__(self, params, from_pth = None,resume_training=False, device=None):
        assert (params['network'] == 'pix2pix')
        super().__init__(params,resume_training,device=device)

        self.network_type = 'pix2pix'
        self.verbose=params['verbose']
        self.conv3d = params['conv3d']
        self.init_feature_kernel = params['init_feature_kernel']
        self.nb_ed_layers = params['nb_ed_layers']
        if "ed_blocks" in params:
            self.ed_blocks = params["ed_blocks"]
        else:
            self.ed_blocks = "conv-relu-norm"
        self.hidden_channels_gen = params['hidden_channels_gen']
        self.hidden_channels_disc = params['hidden_channels_disc']
        self.generator_activation = params['generator_activation']
        self.layer_norm = params['layer_norm']
        self.residual_layer=params['residual_layer']
        self.attention=False if 'attention' not in params else params['attention']

        self.generator_update = params['generator_update']
        self.discriminator_update = params['discriminator_update']

        self.init_model()

        if from_pth:
            if self.verbose>1:
                print('normalement self.load_model(from_pth) mais là non, on le fait juste apres l initialisation des gpus etc')
        else:

            self.init_optimization()
            self.init_losses()

            self.generator_losses = []
            self.discriminator_losses = []
            # self.generator_losses_iter,self.discriminator_losses_iter=[],[]
            self.test_error = []

            self.current_epoch = 1
            self.start_epoch=1

        self.current_iteration = 0
        self.mean_generator_loss = 0
        self.mean_discriminator_loss = 0

        self.amp = self.params['amp']
        if self.amp:
            self.scaler = GradScaler()
        self.autocat_losses= ((not (self.params['adv_loss']=="Wasserstein")) and self.amp)


    def init_model(self):
        if self.verbose>0:
            print(f'models device is supposed to be : {self.device}')
        if self.attention:
            self.Generator = networks_diff.AttentionResUnet(init_dim=self.hidden_channels_gen,out_dim=1,channels=self.input_channels,dim_mults=(1,2,4,8)).to(device = self.device)
        else:
            self.Generator = networks.UNet(input_channel=self.input_channels, ngc = self.hidden_channels_gen,conv3d=self.conv3d,init_feature_kernel=self.init_feature_kernel, nb_ed_layers=self.nb_ed_layers,
                                                output_channel= 1 , generator_activation = self.generator_activation,use_dropout=self.use_dropout, leaky_relu = self.leaky_relu,
                                                norm = self.layer_norm, residual_layer=self.residual_layer, blocks = self.ed_blocks).to(device=self.device)


        self.Discriminator = networks.NEncodingLayers(input_channel=self.input_channels+1,ndc = self.hidden_channels_disc,norm=self.layer_norm,
                                                    output_channel=1,leaky_relu=self.leaky_relu, blocks = self.ed_blocks).to(device=self.device)

        if self.params['jean_zay']:
            helpers_data_parallelism.init_data_parallelism(model=self)

    def init_optimization(self):
        if self.optimizer == 'Adam':
            self.generator_optimizer = optim.Adam(self.Generator.parameters(), lr=self.learning_rate)
            self.discriminator_optimizer = optim.Adam(self.Discriminator.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Unknown optimizer. Choose between : Adam")

        self.learning_rate_policy_infos = self.params['lr_policy']
        if self.learning_rate_policy_infos[0]=='multiplicative':
            mult_rate = self.learning_rate_policy_infos[1]
            self.update_lr_every = self.learning_rate_policy_infos[2]
            lbda = lambda epoch: mult_rate

            self.scheduler_generator = optim.lr_scheduler.MultiplicativeLR(self.generator_optimizer, lbda)
            self.scheduler_discriminator = optim.lr_scheduler.MultiplicativeLR(self.discriminator_optimizer, lbda)


    def init_losses(self):
        self.gp = self.params['with_gradient_penalty']

        self.losses_params = {'adv_loss': self.params['adv_loss'], 'recon_loss': self.params['recon_loss'],
                              'lambda_recon': self.params['lambda_recon'], 'gradient_penalty': self.gp, 'device':self.device}

        self.losses = losses.Pix2PixLosses(self.losses_params)


    def input_data(self, batch):
        self.truePVE = batch[:, 0, :, :, :]
        self.truePVfree = batch[:, -1, 0:1, :, :]

    def forward_D(self):
        ## Update Discriminator
        with torch.no_grad():
            self.DfakePVfree = self.Generator(self.truePVE)

        self.Ddisc_fake_hat = self.Discriminator(self.DfakePVfree.detach(), self.truePVE)
        self.Ddisc_real_hat = self.Discriminator(self.truePVfree, self.truePVE)

    def losses_D(self):
        disc_fake_loss = self.losses.adv_loss(self.Ddisc_fake_hat, self.zeros.expand_as(self.Ddisc_fake_hat))
        disc_real_loss = self.losses.adv_loss(self.Ddisc_real_hat, self.ones.expand_as(self.Ddisc_real_hat))
        self.disc_loss = ((disc_fake_loss + disc_real_loss) / 2)

        if self.gp:
            self.disc_loss += 10 * self.losses.get_gradient_penalty(Discriminator=self.Discriminator,
                                                                    real=self.truePVfree, fake=self.DfakePVfree,
                                                                    condition=self.truePVE)

    def backward_D(self):
        if self.amp:
            self.scaler.scale(self.disc_loss).backward()
            self.scaler.step(self.discriminator_optimizer)
        else:
            self.disc_loss.backward()
            self.discriminator_optimizer.step()

    def forward_G(self):
        self.GfakePVfree = self.Generator(self.truePVE)
        self.Gdisc_fake_hat = self.Discriminator(self.GfakePVfree, self.truePVE)

    def losses_G(self):
        self.gen_loss = self.losses.get_gen_loss(self.Gdisc_fake_hat, self.truePVfree, self.GfakePVfree)

    def backward_G(self):
        if self.amp:
            self.scaler.scale(self.gen_loss).backward()
            self.scaler.step(self.generator_optimizer)
        else:
            self.gen_loss.backward()
            self.generator_optimizer.step()

    def forward(self, batch):
        if  batch.dim()==4:
            self.truePVE = batch
        elif batch.dim()==5:
            self.truePVE = batch[:, 0,:, :, :]

        return self.Generator(self.truePVE)


    def optimize_parameters(self):
        # Discriminator Updates
        self.set_requires_grad(self.Discriminator, requires_grad=True)
        for _ in range(self.discriminator_update):
            self.discriminator_optimizer.zero_grad()
            with autocast(enabled=self.amp):
                self.forward_D()
                with autocast(enabled=self.autocat_losses):
                    self.losses_D()
            self.backward_D()
            if self.amp:
                self.scaler.update()

        # Generator Updates
        self.set_requires_grad(self.Discriminator, requires_grad=False)
        for _ in range(self.generator_update):
            self.generator_optimizer.zero_grad()
            with autocast(enabled=self.amp):
                self.forward_G()
                with autocast(enabled=self.autocat_losses):
                    self.losses_G()
            self.backward_G()
            if self.amp:
                self.scaler.update()

        self.mean_generator_loss+=self.gen_loss.item()
        self.mean_discriminator_loss+=self.disc_loss.item()
        self.current_iteration+=1

    def update_epoch(self):
        self.discriminator_losses.append(self.mean_discriminator_loss / self.current_iteration)
        self.generator_losses.append(self.mean_generator_loss / self.current_iteration)
        if self.verbose > 1:
            print(f'D loss : {round(self.discriminator_losses[-1],5)}')
            print(f'G loss : {round(self.generator_losses[-1],5)}')


        if self.current_epoch % self.update_lr_every ==0:
            self.scheduler_generator.step()
            self.scheduler_discriminator.step()

        if self.verbose > 1:
            print(f'next lr (G): {self.scheduler_generator.get_last_lr()}')

        self.current_epoch+=1
        self.current_iteration=0
        self.mean_generator_loss = 0
        self.mean_discriminator_loss = 0


    def plot_losses(self, save, wait, title):
        plots.plot_losses_double_model(self.generator_losses, self.discriminator_losses, self.test_error,labels=['Generator Loss','Discriminator Loss'], save=save, wait = wait, title = title)

    def save_model(self, output_path=None, save_json=False):
        self.params['start_epoch'] = self.start_epoch
        self.params['current_epoch'] = self.current_epoch
        jean_zay = self.params['jean_zay']

        if not output_path:
            if self.output_folder:
                output_path = os.path.join(self.output_folder, self.output_pth)
            else:
                raise ValueError("Error: no output_folder specified")
        if jean_zay:
            torch.save({'saving_date': time.asctime(),
                        'epoch': self.current_epoch,
                        'gen': self.Generator.module.state_dict(),
                        'gen_opt': self.generator_optimizer.state_dict(),
                        'disc': self.Discriminator.module.state_dict(),
                        'disc_opt': self.discriminator_optimizer.state_dict(),
                        'gen_losses': self.generator_losses,
                        'disc_losses': self.discriminator_losses,
                        'test_error': self.test_error,
                        'params': self.params
                        }, output_path )
        else:
            torch.save({'saving_date': time.asctime(),
                        'epoch': self.current_epoch,
                        'gen': self.Generator.state_dict(),
                        'gen_opt': self.generator_optimizer.state_dict(),
                        'disc': self.Discriminator.state_dict(),
                        'disc_opt': self.discriminator_optimizer.state_dict(),
                        'gen_losses': self.generator_losses,
                        'disc_losses': self.discriminator_losses,
                        'test_error': self.test_error,
                        'params': self.params
                        }, output_path )
        if self.verbose > 0:
            print(f'Model saved at : {output_path}')

        if save_json:
            output_json = output_path[:-4]+'.json'
            formatted_params = self.format_params()
            jsonFile = open(output_json, "w")
            jsonFile.write(formatted_params)
            jsonFile.close()

    def load_model(self,pth_path):
        if self.verbose > 0:
            print(f'Loading Model from {pth_path}... ')
        checkpoint = torch.load(pth_path, map_location=self.device)

        if hasattr(self.Generator, 'module'):
            self.Generator.module.load_state_dict(checkpoint['gen'])
            self.Discriminator.module.load_state_dict(checkpoint['disc'])
        else:
            self.Generator.load_state_dict(checkpoint['gen'])
            self.Discriminator.load_state_dict(checkpoint['disc'])
        self.generator_losses = checkpoint['gen_losses']
        self.discriminator_losses = checkpoint['disc_losses']
        self.test_error = checkpoint['test_error']
        self.current_epoch = checkpoint['epoch']

        if self.resume_training:
            self.init_optimization()
            self.init_losses()
            self.generator_optimizer.load_state_dict(checkpoint['gen_opt'])
            self.discriminator_optimizer.load_state_dict(checkpoint['disc_opt'])

            self.start_epoch=self.current_epoch

    def switch_eval(self):
        self.Generator.eval()
        self.Discriminator.eval()

    def switch_train(self):
        self.Generator.train()
        self.Discriminator.train()

    def switch_device(self, device):
        self.device = device
        self.ones = self.ones.to(device=device)
        self.zeros = self.zeros.to(device=device)
        self.Generator = self.Generator.to(device=device)
        self.Discriminator = self.Discriminator.to(device=device)
        if hasattr(self, "losses"):
            self.losses.ones = self.losses.ones.to(device=device)
            if hasattr(self.losses, "gradient_penalty"):
                self.losses.gradient_penalty.switch_device(device=device)

    def show_infos(self, mse = False):
        formatted_params = self.format_params()
        print('*'*80)
        print('PARAMETRES (json param file) : \n')
        print(formatted_params)
        print('*' * 80)

        print('MEAN SQUARE ERROR on TEST DATA')
        print(f'list of errors on test data :{self.test_error} ')
        print('*' * 80)
        print(self.Generator)
        print(self.Discriminator)
        print('*' * 80)
        if hasattr(self, "losses"):
            print('Losses : ')
            print(self.losses_params)
            print(self.losses)
            print('*' * 80)

        # helpers_params.make_and_print_params_info_table([self.params])
