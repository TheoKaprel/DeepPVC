import os.path
import time
import torch
from torch import optim
import json
import copy
from abc import ABC, abstractmethod

from . import networks, losses,plots, helpers, helpers_params



class Models(ABC):
    def __init__(self,  params,is_resume):
        self.is_resume = is_resume
        self.params = params
        self.device = helpers.get_auto_device(self.params['device'])

        self.n_epochs = params['n_epochs']
        self.input_channels = params['input_channels']
        self.nb_ed_layers = params['nb_ed_layers']
        self.hidden_channels_gen = params['hidden_channels_gen']
        self.hidden_channels_disc = params['hidden_channels_disc']
        self.generator_activation = params['generator_activation']
        self.generator_norm = params['generator_norm']
        self.use_dropout = params['use_dropout']
        self.sum_norm = params['sum_norm']

        if self.generator_activation=='relu_min':
            norm = self.params['norm']
            self.vmin = -norm[0]/norm[1]
        else:
            self.vmin = None

        self.learning_rate = params['learning_rate']
        self.generator_update = params['generator_update']
        self.discriminator_update = params['discriminator_update']
        self.optimizer = params['optimizer']


        self.output_folder = self.params['output_folder']
        self.output_pth = self.params['output_pth']

    def input_data(self, batch):
        self.truePVE = batch[:, 0,:, :, :].to(self.device).float()
        self.truePVfree = batch[:, 1,:, :, :].to(self.device).float()

    def format_params(self):
        formatted_params = copy.deepcopy(self.params)
        listnorm = formatted_params['norm']
        formatted_params['norm'] = str(listnorm)
        return json.dumps(formatted_params, indent=4)

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def init_optimization(self):
        pass

    @abstractmethod
    def init_losses(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    @abstractmethod
    def update_epoch(self):
        pass

    @abstractmethod
    def plot_losses(self, save, wait, title):
        pass

    @abstractmethod
    def save_model(self, output_path=None, save_json=False):
        pass

    @abstractmethod
    def load_model(self, pth_path):
        pass

    @abstractmethod
    def switch_eval(self):
        pass

    @abstractmethod
    def switch_train(self):
        pass

    @abstractmethod
    def switch_device(self, device):
        pass



class Pix2PixModel(Models):
    def __init__(self, params, is_resume, pth=None, eval=False):
        super().__init__(params, is_resume)

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

            self.current_epoch = 1
            self.start_epoch=1

        self.current_iteration = 0
        self.mean_generator_loss = 0
        self.mean_discriminator_loss = 0


        if eval:
            self.switch_eval()
        else:
            self.switch_train()


    def init_model(self):
        self.Generator = networks.UNetGenerator(input_channel=self.input_channels, ngc = self.hidden_channels_gen, nb_ed_layers=self.nb_ed_layers,
                                                output_channel=self.input_channels,generator_activation = self.generator_activation,use_dropout=self.use_dropout,
                                                sum_norm = self.sum_norm,norm = self.generator_norm, vmin=self.vmin).to(device=self.device)

        self.Discriminator = networks.NLayerDiscriminator(input_channel=2*self.input_channels,
                                                          ndc = self.hidden_channels_disc,
                                                          output_channel=self.input_channels).to(device=self.device)

    def init_optimization(self):
        if self.optimizer == 'Adam':
            self.generator_optimizer = optim.Adam(self.Generator.parameters(), lr=self.learning_rate)
            self.discriminator_optimizer = optim.Adam(self.Discriminator.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Unknown optimizer. Choose between : Adam")

    def init_losses(self):
        self.losses_params = {'adv_loss': self.params['adv_loss'], 'recon_loss': self.params['recon_loss'], 'lambda_recon': self.params['lambda_recon']}

        self.losses = losses.Pix2PixLosses(self.losses_params)

    def forward_D(self):
        ## Update Discriminator
        with torch.no_grad():
            self.DfakePVfree = self.Generator(self.truePVE)

        self.Ddisc_fake_hat = self.Discriminator(self.DfakePVfree.detach(), self.truePVE)
        self.Ddisc_real_hat = self.Discriminator(self.truePVfree, self.truePVE)

    def backward_D(self):
        disc_fake_loss = self.losses.adv_loss(self.Ddisc_fake_hat, torch.zeros_like(self.Ddisc_fake_hat))
        disc_real_loss = self.losses.adv_loss(self.Ddisc_real_hat, torch.ones_like(self.Ddisc_real_hat))
        self.disc_loss = ((disc_fake_loss + disc_real_loss) / 2)

        self.disc_loss.backward(retain_graph=True)
        self.discriminator_optimizer.step()

    def forward_G(self):
        self.GfakePVfree = self.Generator(self.truePVE)
        self.Gdisc_fake_hat = self.Discriminator(self.GfakePVfree, self.truePVE)

    def backward_G(self):
        self.gen_loss = self.losses.get_gen_loss(self.Gdisc_fake_hat, self.truePVfree, self.GfakePVfree)
        self.gen_loss.backward()
        self.generator_optimizer.step()

    def optimize_parameters(self):
        # Discriminator Updates
        for _ in range(self.discriminator_update):
            self.discriminator_optimizer.zero_grad()
            self.forward_D()
            self.backward_D()

        # Generator Updates
        for _ in range(self.generator_update):
            self.generator_optimizer.zero_grad()
            self.forward_G()
            self.backward_G()

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

    def plot_losses(self, save, wait, title):
        plots.plot_losses(self.discriminator_losses, self.generator_losses, self.test_mse, save=save, wait = wait, title = title)

    def save_model(self, output_path=None, save_json=False):
        self.params['start_epoch'] = self.start_epoch
        self.params['current_epoch'] = self.current_epoch

        if not output_path:
            if self.output_folder:
                output_path = os.path.join(self.output_folder, self.output_pth)
            else:
                raise ValueError("Error: no output_folder specified")
        else:
            raise ValueError("Error: no output_path")

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
                    }, output_path )
        print(f'Model saved at : {output_path}')

        if save_json:
            output_json = output_path[:-4]+'.json'
            formatted_params = self.format_params()
            jsonFile = open(output_json, "w")
            jsonFile.write(formatted_params)
            jsonFile.close()

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

    def show_infos(self, mse = False):
        formatted_params = self.format_params()
        print('*'*80)
        print('PARAMETRES (json param file) : \n')
        print(formatted_params)
        print('*' * 80)

        print('MEAN SQUARE ERROR on TEST DATA')
        print(f'list of MSE on test data :{self.test_mse} ')
        print('*' * 80)
        print(self.Generator)
        print(self.Discriminator)
        print('*' * 80)

        helpers_params.make_and_print_params_info_table([self.params])