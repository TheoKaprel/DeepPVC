import time

import torch
from torch import optim
from . import networks
from functions import losses
from utils import plots
from utils import helpers
import os
import json

class PVEPix2PixModel():
    def __init__(self, params,is_resume,pth=None, eval=False):

        self.is_resume = is_resume
        self.params = params
        self.device = helpers.get_auto_device(self.params['device'])
        self.output_path = self.params['output_path']

        if self.is_resume:
            if pth is None:
                pth = self.params['start_pth'][-1]
            self.load_model(pth)
        else:
            self.init_model()
            self.init_optimization()
            self.init_losses()

            self.generator_losses = []
            self.discriminator_losses = []
            self.test_mse = []

            self.current_epoch = 0
            self.start_epoch=0

        self.current_iteration = 0
        self.mean_generator_loss = 0
        self.mean_discriminator_loss = 0


        if eval:
            self.switch_eval()
        else:
            self.switch_train()

    def init_model(self):
        params = self.params

        self.input_channels = params['input_channels']
        self.hidden_channels_gen = params['hidden_channels_gen']
        self.hidden_channels_disc = params['hidden_channels_disc']
        self.generator_activation = params['generator_activation']

        self.Generator = networks.UNetGenerator(input_channel=self.input_channels,
                                                ngc = self.hidden_channels_gen,
                                                output_channel=self.input_channels,generator_activation = self.generator_activation, norm = True).to(device=self.device)

        self.Discriminator = networks.NLayerDiscriminator(input_channel=2*self.input_channels,
                                                          ndc = self.hidden_channels_disc,
                                                          output_channel=self.input_channels).to(device=self.device)

    def init_optimization(self):
        params = self.params
        self.n_epochs = params['n_epochs']

        self.learning_rate = params['learning_rate']
        self.generator_update = params['generator_update']
        self.discriminator_update = params['discriminator_update']
        self.optimizer = params['optimizer']

        if self.optimizer == 'Adam':
            self.generator_optimizer = optim.Adam(self.Generator.parameters(), lr=self.learning_rate)
            self.discriminator_optimizer = optim.Adam(self.Discriminator.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Unknown optimizer. Choose between : Adam")

    def init_losses(self):
        self.losses_params = {'adv_loss': self.params['adv_loss'], 'recon_loss': self.params['recon_loss'], 'lambda_recon': self.params['lambda_recon']}

        self.losses = losses.Pix2PixLosses(self.losses_params)

    def input_data(self, batch):
        self.truePVE = batch[:, 0, :, :].unsqueeze(1).to(self.device)
        self.truePVfree = batch[:, 1, :, :].unsqueeze(1).to(self.device)

    def forward(self):
        ## Update Discriminator
        with torch.no_grad():
            self.fakePVfree = self.Generator(self.truePVE.float())

        self.disc_fake_hat = self.Discriminator(self.fakePVfree.detach().float(), self.truePVE.float())
        self.disc_real_hat = self.Discriminator(self.truePVfree.float(), self.truePVE.float())

    def backward_D(self, back=True):
        disc_fake_loss = self.losses.adv_loss(self.disc_fake_hat, torch.zeros_like(self.disc_fake_hat))
        disc_real_loss = self.losses.adv_loss(self.disc_real_hat, torch.ones_like(self.disc_real_hat))
        self.disc_loss = ((disc_fake_loss + disc_real_loss) / 2)
        if back:
            self.disc_loss.backward(retain_graph=True)
            self.discriminator_optimizer.step()

    def backward_G(self, back=True):
        ## Update Generator

        self.gen_loss = self.losses.get_gen_loss(self.Generator, self.Discriminator, self.truePVfree, self.truePVE)
        if back:
            self.gen_loss.backward()
            self.generator_optimizer.step()

    def optimize_parameters(self):

        # Discriminator Updates
        for _ in range(self.discriminator_update):
            self.discriminator_optimizer.zero_grad()
            self.forward()
            self.backward_D(back=True)

        # Generator Updates
        for _ in range(self.generator_update):
            self.generator_optimizer.zero_grad()
            self.backward_G(back=True)

        self.mean_generator_loss+=self.gen_loss.item()
        self.mean_discriminator_loss+=self.disc_loss.item()

        self.current_iteration+=1

    def update_epoch(self):
        self.discriminator_losses.append(self.mean_discriminator_loss / self.current_iteration)
        self.generator_losses.append(self.mean_generator_loss / self.current_iteration)

        self.current_epoch+=1
        self.current_iteration=0
        self.mean_generator_loss = 0
        self.mean_discriminator_loss = 0

    def plot_losses(self, save):
        plots.plot_losses(self.discriminator_losses, self.generator_losses, self.test_mse, save=save)

    def save_model(self):
        self.params['start_epoch'] = self.start_epoch
        self.params['current_epoch'] = self.current_epoch

        torch.save({'saving_date': time.asctime(),
                    'epoch': self.current_epoch,
                    'gen': self.Generator.state_dict(),
                    'gen_opt': self.generator_optimizer.state_dict(),
                    'disc': self.Discriminator.state_dict(),
                    'disc_opt': self.discriminator_optimizer.state_dict(),
                    'gen_losses': self.generator_losses,
                    'disc_losses': self.discriminator_losses,
                    'test_mse': self.test_mse,
                    'params': self.params
                    }, self.output_path )
        print(f'Model saved at : {self.output_path}')

    def load_model(self,pth_path):

        print(f'Loading Model from {pth_path}... ')
        checkpoint = torch.load(pth_path, map_location=self.device)

        self.init_model()
        self.init_optimization()
        self.init_losses()

        self.Generator.load_state_dict(checkpoint['gen'])
        self.Discriminator.load_state_dict(checkpoint['disc'])


        self.generator_optimizer.load_state_dict(checkpoint['gen_opt'])
        self.discriminator_optimizer.load_state_dict(checkpoint['disc_opt'])

        self.generator_losses = checkpoint['gen_losses']
        self.discriminator_losses = checkpoint['disc_losses']
        self.test_mse = checkpoint['test_mse']
        self.current_epoch = checkpoint['epoch']
        self.start_epoch=self.current_epoch

    def switch_eval(self):
        self.Generator.eval()
        self.Discriminator.eval()

    def switch_train(self):
        self.Generator.train()
        self.Discriminator.train()

    def switch_device(self, device):
        self.device = device
        self.Generator.to(device=device)
        self.Discriminator.to(device=device)

    def test(self, img):
        with torch.no_grad():
            output = self.Generator(img.float())
        return output

    def show_infos(self):
        strparams = self.params
        strparams['norm'] = str(strparams['norm'])
        print('*'*80)
        print('PARAMETRES (json param file) : \n')
        json_formatted_str = json.dumps(strparams, indent=4)
        print(json_formatted_str)
        print('*' * 80)
        print('MEAN SQUARE ERROR on TEST DATA')

        print(f'list of MSE on test data :{self.test_mse} ')
        print('*' * 80)
        print(self.Generator)
        print(self.Discriminator)
        print('*' * 80)

