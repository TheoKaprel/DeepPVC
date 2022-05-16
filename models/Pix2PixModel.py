import torch
from torch import optim
from . import networks
from functions import losses
from utils import plots
import os
import json

class PVEPix2PixModel():
    def __init__(self, params, eval=False):
        # if not load_pth:
        #     if training_params and (not losses_params):
        #         raise ValueError("You have to specify training and loss parameters (losses_params is missing)")
        #     if (not training_params) and (losses_params):
        #         raise ValueError("You have to specify training and loss parameters (training_params is missing)")
        #     if (not training_params) and (not losses_params):
        #         raise ValueError("You have to specify either training and loss parameters (training_params and losses_params) either a pth file to load")
        # else:
        #     if training_params or losses_params:
        #         print(f"WARNING : The training and loss parameters will be infered from the given load_pth : {load_pth} and not from the given training_params nor losses_params")

        self.params = params

        if 'start_pth' in self.params and self.params['start_pth'] is not None:
            self.load_model(self.params['start_pth'])
        else:
            self.init_model()
            self.init_optimization()
            self.init_losses()

            self.generator_losses = []
            self.discriminator_losses = []
            self.current_epoch = 0


        self.current_iteration = 0
        self.mean_discriminator_loss = 0
        self.mean_generator_loss = 0

        self.display_step = self.params['display_step']
        self.device = torch.device(self.params['device'])

        if eval:
            self.swith_eval()

    def init_model(self):
        params = self.params

        self.input_channels = params['input_channels']
        self.hidden_channels_gen = params['hidden_channels_gen']
        self.hidden_channels_disc = params['hidden_channels_disc']

        self.Generator = networks.UNetGenerator(input_channel=self.input_channels,
                                                ngc = self.hidden_channels_gen,
                                                output_channel=self.input_channels)

        self.Discriminator = networks.NLayerDiscriminator(input_channel=2*self.input_channels,
                                                          ndc = self.hidden_channels_disc,
                                                          output_channel=self.input_channels)

    def init_optimization(self):
        params = self.params
        self.n_epochs = params['n_epochs']

        self.learning_rate = params['learning_rate']
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

    def backward_D(self):
        disc_fake_loss = self.losses.adv_loss(self.disc_fake_hat, torch.zeros_like(self.disc_fake_hat))
        print('disc fake loss')
        disc_real_loss = self.losses.adv_loss(self.disc_real_hat, torch.ones_like(self.disc_real_hat))
        print('disc eal loss')
        self.disc_loss = ((disc_fake_loss + disc_real_loss) / 2)
        self.disc_loss.backward(retain_graph=True)
        print('disc loss back')
        self.discriminator_optimizer.step()
        print('step')

    def backward_G(self):
        ## Update Generator

        self.gen_loss = self.losses.get_gen_loss(self.Generator, self.Discriminator, self.truePVfree, self.truePVE)
        self.gen_loss.backward()
        self.generator_optimizer.step()

    def optimize_parameters(self):
        self.discriminator_optimizer.zero_grad()
        self.generator_optimizer.zero_grad()

        self.forward()
        self.backward_D()
        self.backward_G()

        # Keep track of the average discriminator loss
        self.mean_discriminator_loss += self.disc_loss.item() / self.display_step
        # Keep track of the average generator loss
        self.mean_generator_loss += self.gen_loss.item() / self.display_step
        print('mean losses')
        self.discriminator_losses.append(self.disc_loss.item())
        self.generator_losses.append(self.gen_loss.item())


        self.current_iteration+=1

    def display(self):
        ### Visualization code ###
        print(f"Epoch {self.current_epoch}: Step {self.current_iteration-1}: Generator loss: {self.mean_generator_loss}, Discriminator loss: {self.mean_discriminator_loss}")

        plots.show_tensor_images(torch.cat((self.truePVE, self.truePVfree, self.fakePVfree), 1))

        self.mean_generator_loss = 0
        self.mean_discriminator_loss = 0


    def update_epoch(self):
        self.current_epoch+=1
        self.current_iteration=0

    def plot_losses(self):
        plots.plot_losses(self.discriminator_losses, self.generator_losses)

    def save_model(self,folder, extention=None):
        if extention:
            output_filename = f"pix2pix_{extention}_{self.current_epoch}.pth"
        else:
            output_filename = f"pix2pix_{self.current_epoch}.pth"

        output_path = os.path.join(folder, output_filename)


        torch.save({'epoch': self.current_epoch,
                    'gen': self.Generator.state_dict(),
                    'gen_opt': self.generator_optimizer.state_dict(),
                    'disc': self.Discriminator.state_dict(),
                    'disc_opt': self.discriminator_optimizer.state_dict(),
                    'gen_losses': self.generator_losses,
                    'disc_losses': self.discriminator_losses,
                    'params': self.params
                    }, output_path)

    def load_model(self,pth_path):
        checkpoint = torch.load(pth_path)
        self.params = checkpoint['params']
        self.init_model()
        self.init_optimization()
        self.init_losses()

        self.Generator.load_state_dict(checkpoint['gen'])
        self.Discriminator.load_state_dict(checkpoint['disc'])


        self.generator_optimizer = checkpoint['gen_opt']
        self.discriminator_optimizer = checkpoint['disc_opt']

        self.generator_losses = checkpoint['gen_losses']
        self.discriminator_losses = checkpoint['disc_losses']
        self.current_epoch = checkpoint['epoch']

    def swith_eval(self):
        self.Generator.eval()
        self.Discriminator.eval()

    def show_infos(self):
        print('*'*80)
        print('PARAMETRES (json param file) : \n')
        json_formatted_str = json.dumps(self.params, indent=4)
        print(json_formatted_str)
        print(f'Le Generateur ressemble à ça : {self.Generator}')

