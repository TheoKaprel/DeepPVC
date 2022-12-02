import os.path
import time
import torch
from torch import optim
import json
import copy
from abc import abstractmethod

from . import networks, losses,plots, helpers, helpers_params

class ModelInstance():
    def __new__(cls, params, from_pth = None):
        network_architecture = params['network']
        if network_architecture == 'pix2pix':
            return Pix2PixModel(params=params, from_pth=from_pth)
        elif network_architecture == 'unet':
            return UNetModel(params=params, from_pth=from_pth)
        elif network_architecture == 'denoiser_pvc':
            return UNet_Denoiser_PVC(params=params, from_pth=from_pth)
        else:
            print(f"ERROR : unknown network architecture ({network_architecture})")
            exit(0)


class ModelBase():
    def __init__(self,  params):
        self.params = params
        self.device = helpers.get_auto_device(self.params['device'])

        self.n_epochs = params['n_epochs']
        self.learning_rate = params['learning_rate']
        self.input_channels = params['input_channels']

        self.use_dropout = params['use_dropout']
        self.sum_norm = params['sum_norm']
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
    def forward(self, batch):
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



class Pix2PixModel(ModelBase):
    def __init__(self, params, from_pth = None):
        assert (params['network'] == 'pix2pix')
        super().__init__(params)

        self.nb_ed_layers = params['nb_ed_layers']
        self.hidden_channels_gen = params['hidden_channels_gen']
        self.hidden_channels_disc = params['hidden_channels_disc']
        self.generator_activation = params['generator_activation']
        self.generator_norm = params['generator_norm']
        if self.generator_activation=='relu_min':
            norm = self.params['norm']
            self.vmin = -norm[0]/norm[1]
        else:
            self.vmin = None

        self.generator_update = params['generator_update']
        self.discriminator_update = params['discriminator_update']

        if from_pth:
            self.load_model(from_pth)
        else:
            self.init_model()
            self.init_optimization()
            self.init_losses()

            self.generator_losses = []
            self.discriminator_losses = []
            self.test_error = []

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

        self.learning_rate_policy_infos = self.params['lr_policy']
        if self.learning_rate_policy_infos[0]=='multiplicative':
            mult_rate = self.learning_rate_policy_infos[1]
            lbda = lambda epoch: mult_rate

            self.scheduler_generator = optim.lr_scheduler.MultiplicativeLR(self.generator_optimizer, lbda)
            self.scheduler_discriminator = optim.lr_scheduler.MultiplicativeLR(self.discriminator_optimizer, lbda)


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

    def forward(self, batch):
        if  batch.dim()==4:
            self.truePVE = batch.to(self.device).float()
        elif (batch.dim()==5) and (batch.shape[1]==2):
            self.input_data(batch=batch)
        elif batch.dim()==5 and (batch.shape[1]==1):
            self.truePVE = batch[:, 0,:, :, :].to(self.device).float()

        fakePVfree = self.Generator(self.truePVE)
        return fakePVfree

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

        self.scheduler_generator.step()
        self.scheduler_discriminator.step()

    def plot_losses(self, save, wait, title):
        plots.plot_losses_double_model(self.generator_losses, self.discriminator_losses, self.test_error,labels=['Generator Loss','Discriminator Loss'], save=save, wait = wait, title = title)

    def save_model(self, output_path=None, save_json=False):
        self.params['start_epoch'] = self.start_epoch
        self.params['current_epoch'] = self.current_epoch

        if not output_path:
            if self.output_folder:
                output_path = os.path.join(self.output_folder, self.output_pth)
            else:
                raise ValueError("Error: no output_folder specified")

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
        self.test_error = checkpoint['test_error']
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
        print(f'list of errors on test data :{self.test_error} ')
        print('*' * 80)
        print(self.Generator)
        print(self.Discriminator)
        print('*' * 80)

        helpers_params.make_and_print_params_info_table([self.params])


class UNetModel(ModelBase):
    def __init__(self, params, from_pth = None):
        assert (params['network'] == 'unet')
        super().__init__(params)

        self.nb_ed_layers = params['nb_ed_layers']
        self.hidden_channels_unet = params['hidden_channels_unet']
        self.unet_activation = params['unet_activation']
        self.unet_norm = params['unet_norm']
        if self.unet_activation=='relu_min':
            norm = self.params['norm']
            self.vmin = -norm[0]/norm[1]
        else:
            self.vmin = None

        if from_pth:
            self.load_model(from_pth)
        else:
            self.init_model()
            self.init_optimization()
            self.init_losses()

            self.unet_losses = []
            self.test_error = []

            self.current_epoch = 1
            self.start_epoch=1

        self.current_iteration = 0
        self.mean_unetlosses = 0


        if eval:
            self.switch_eval()
        else:
            self.switch_train()


    def init_model(self):
        self.UNet = networks.UNetGenerator(input_channel=self.input_channels, ngc = self.hidden_channels_unet, nb_ed_layers=self.nb_ed_layers,
                                                output_channel=self.input_channels,generator_activation = self.unet_activation,use_dropout=self.use_dropout,
                                                sum_norm = self.sum_norm,norm = self.unet_norm, vmin=self.vmin).to(device=self.device)

    def init_optimization(self):
        if self.optimizer == 'Adam':
            self.unet_optimizer = optim.Adam(self.UNet.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Unknown optimizer. Choose between : Adam")

        self.learning_rate_policy_infos = self.params['lr_policy']
        if self.learning_rate_policy_infos[0]=='multiplicative':
            mult_rate = self.learning_rate_policy_infos[1]
            lbda = lambda epoch: mult_rate

            self.scheduler_unet = optim.lr_scheduler.MultiplicativeLR(self.unet_optimizer, lbda)



    def init_losses(self):
        self.losses_params = {'recon_loss': self.params['recon_loss']}

        # self.losses = losses.Pix2PixLosses(self.losses_params)
        self.losses = losses.UNetLosses(self.losses_params)

    def forward_UNet(self):
        self.fakePVfree = self.UNet(self.truePVE)

    def backward_UNet(self):
        self.unet_loss = self.losses.get_unet_loss(self.truePVfree, self.fakePVfree)
        self.unet_loss.backward()
        self.unet_optimizer.step()

    def forward(self, batch):
        if batch.dim()==4:
            self.truePVE = batch.to(self.device).float()
        elif batch.dim()==5:
            if (batch.shape[1] == 1 or batch.shape[1] == 2):
                self.truePVE = batch[:,0,:,:,:].to(self.device).float()
            elif batch.shape[1] == 3:
                self.truePVE = batch[:,0,:,:,:].to(self.device).float()


        fakePVfree = self.UNet(self.truePVE)
        return fakePVfree

    def optimize_parameters(self):
        # UNET Updats

        self.unet_optimizer.zero_grad()
        self.forward_UNet()
        self.backward_UNet()

        self.mean_unetlosses+=self.unet_loss.item()

        self.current_iteration+=1

    def update_epoch(self):
        self.unet_losses.append(self.mean_unetlosses / self.current_iteration)

        self.current_epoch+=1
        self.current_iteration=0
        self.mean_unetlosses = 0

        self.scheduler_unet.step()

    def plot_losses(self, save, wait, title):
        plots.plot_losses_UNet(self.unet_losses, self.test_error, save=save, wait = wait, title = title)

    def save_model(self, output_path=None, save_json=False):
        self.params['start_epoch'] = self.start_epoch
        self.params['current_epoch'] = self.current_epoch

        if not output_path:
            if self.output_folder:
                output_path = os.path.join(self.output_folder, self.output_pth)
            else:
                raise ValueError("Error: no output_folder specified")

        torch.save({'saving_date': time.asctime(),
                    'epoch': self.current_epoch,
                    'unet': self.UNet.state_dict(),
                    'unet_opt': self.unet_optimizer.state_dict(),
                    'unet_losses': self.unet_losses,
                    'test_error': self.test_error,
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

        self.UNet.load_state_dict(checkpoint['unet'])


        self.unet_optimizer.load_state_dict(checkpoint['unet_opt'])

        self.unet_losses = checkpoint['unet_losses']
        self.test_error = checkpoint['test_error']
        self.current_epoch = checkpoint['epoch']
        self.start_epoch=self.current_epoch

    def switch_eval(self):
        self.UNet.eval()

    def switch_train(self):
        self.UNet.train()

    def switch_device(self, device):
        self.device = device
        self.UNet.to(device=device)

    def show_infos(self, mse = False):
        formatted_params = self.format_params()
        print('*'*80)
        print('PARAMETRES (json param file) : \n')
        print(formatted_params)
        print('*' * 80)

        print('MEAN SQUARE ERROR on TEST DATA')
        print(f'list of MSE on test data :{self.test_error} ')
        print('*' * 80)
        print(self.UNet)
        print('*' * 80)

        # helpers_params.make_and_print_params_info_table([self.params])




class UNet_Denoiser_PVC(ModelBase):
    def __init__(self, params, from_pth = None):
        assert(params['network']=='denoiser_pvc')
        super().__init__(params)

        self.nb_ed_layers_denoiser = params['nb_ed_layers_denoiser']
        self.hidden_channels_unet_denoiser = params['hidden_channels_unet_denoiser']
        self.unet_denoiser_activation = params['unet_denoiser_activation']
        self.unet_denoiser_norm = params['unet_denoiser_norm']

        self.nb_ed_layers_pvc = params['nb_ed_layers_pvc']
        self.hidden_channels_unet_pvc = params['hidden_channels_unet_pvc']
        self.unet_pvc_activation = params['unet_pvc_activation']
        self.unet_pvc_norm = params['unet_pvc_norm']

        if (self.unet_denoiser_activation=='relu_min' or self.unet_pvc_activation=='relu_min'):
            norm = self.params['norm']
            self.vmin = -norm[0]/norm[1]
        else:
            self.vmin = None


        if from_pth:
            self.load_model(from_pth)
        else:
            self.init_model()
            self.init_optimization()
            self.init_losses()

            self.unet_denoiser_list_losses = []
            self.unet_pvc_list_losses = []
            self.test_error = []

            self.current_epoch = 1
            self.start_epoch=1

        self.current_iteration = 0
        self.mean_unet_denoiser_losses = 0
        self.mean_unet_pvc_losses = 0


        if eval:
            self.switch_eval()
        else:
            self.switch_train()


    def init_model(self):
        self.UNet_denoiser = networks.UNetGenerator(input_channel=self.input_channels, ngc = self.hidden_channels_unet_denoiser, nb_ed_layers=self.nb_ed_layers_denoiser,
                                                output_channel=self.input_channels,generator_activation = self.unet_denoiser_activation,use_dropout=self.use_dropout,
                                                sum_norm = self.sum_norm,norm = self.unet_denoiser_norm, vmin=self.vmin).to(device=self.device)

        self.UNet_pvc = networks.UNetGenerator(input_channel=self.input_channels, ngc = self.hidden_channels_unet_pvc, nb_ed_layers=self.nb_ed_layers_pvc,
                                                output_channel=self.input_channels,generator_activation = self.unet_pvc_activation,use_dropout=self.use_dropout,
                                                sum_norm = self.sum_norm,norm = self.unet_pvc_norm, vmin=self.vmin).to(device=self.device)

    def init_optimization(self):
        self.denoiser_update = self.params['denoiser_update']
        self.pvc_update = self.params['pvc_update']
        self.denoiser_loss = torch.Tensor([0])
        self.pvc_loss = torch.Tensor([0])

        if self.optimizer == 'Adam':
            self.unet_denoiser_optimizer = optim.Adam(self.UNet_denoiser.parameters(), lr=self.learning_rate)
            self.unet_pvc_optimizer = optim.Adam(self.UNet_pvc.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Unknown optimizer. Choose between : Adam")

        self.learning_rate_policy_infos = self.params['lr_policy']
        if self.learning_rate_policy_infos[0]=='multiplicative':
            mult_rate = self.learning_rate_policy_infos[1]
            lbda = lambda epoch: mult_rate

            self.scheduler_pvc = optim.lr_scheduler.MultiplicativeLR(self.unet_pvc_optimizer, lbda)
            self.scheduler_denoiser = optim.lr_scheduler.MultiplicativeLR(self.unet_denoiser_optimizer, lbda)



    def init_losses(self):
        self.denoiser_losses_params = {'recon_loss': self.params['recon_loss_denoiser']}
        self.pvc_losses_params = {'recon_loss': self.params['recon_loss_pvc']}

        self.unet_denoiser_losses = losses.UNetLosses(self.denoiser_losses_params)
        self.unet_pvc_losses = losses.UNetLosses(self.pvc_losses_params)

    def input_data(self, batch):
        self.noisyPVE = batch[:, 0,:, :, :].to(self.device).float()
        self.truePVE = batch[:, 1,:, :, :].to(self.device).float()
        self.truePVfree = batch[:, 2,:, :, :].to(self.device).float()


    def forward_denoiser(self):
        self.fakePVE = self.UNet_denoiser(self.noisyPVE)

    def forward_pvc(self):
        self.fakePVE = self.UNet_denoiser(self.noisyPVE)
        self.fakePVfree = self.UNet_pvc(self.fakePVE)

    def backward_denoiser(self):
        self.denoiser_loss = self.unet_denoiser_losses.get_unet_loss(self.truePVE, self.fakePVE)
        self.denoiser_loss.backward()
        self.unet_denoiser_optimizer.step()

    def backward_pvc(self):
        self.pvc_loss = self.unet_pvc_losses.get_unet_loss(self.truePVfree, self.fakePVfree)
        self.pvc_loss.backward()
        self.unet_pvc_optimizer.step()


    def forward(self, batch):
        if batch.dim()==4:
            self.noisyPVE = batch.to(self.device).float()
        elif batch.dim()==5:
            self.noisyPVE = batch[:,0,:,:,:].to(self.device).float()

        denoisedPVE = self.UNet_denoiser(self.noisyPVE)
        fakePVfree = self.UNet_pvc(denoisedPVE)
        return fakePVfree

    def optimize_parameters(self):
        # denoiser update
        for _ in range(self.denoiser_update):
            self.unet_denoiser_optimizer.zero_grad()
            self.forward_denoiser()
            self.backward_denoiser()
        self.mean_unet_denoiser_losses+=self.denoiser_loss.item()

        # pvc update
        for _ in range(self.pvc_update):
            self.unet_pvc_optimizer.zero_grad()
            self.forward_pvc()
            self.backward_pvc()
        self.mean_unet_pvc_losses+=self.pvc_loss.item()

        self.current_iteration+=1

    def update_epoch(self):
        self.unet_denoiser_list_losses.append(self.mean_unet_denoiser_losses / self.current_iteration)
        self.unet_pvc_list_losses.append(self.mean_unet_pvc_losses / self.current_iteration)

        self.current_epoch+=1
        self.current_iteration=0
        self.mean_unet_denoiser_losses = 0
        self.mean_unet_pvc_losses = 0

        self.scheduler_pvc.step()
        self.scheduler_denoiser.step()

    def plot_losses(self, save, wait, title):
        plots.plot_losses_double_model(self.unet_denoiser_list_losses, self.unet_pvc_list_losses, self.test_error,labels=['Denoiser Loss','DeepPVC Loss'], save=save, wait = wait, title = title)

    def save_model(self, output_path=None, save_json=False):
        self.params['start_epoch'] = self.start_epoch
        self.params['current_epoch'] = self.current_epoch

        if not output_path:
            if self.output_folder:
                output_path = os.path.join(self.output_folder, self.output_pth)
            else:
                raise ValueError("Error: no output_folder specified")

        torch.save({'saving_date': time.asctime(),
                    'epoch': self.current_epoch,
                    'unet_denoiser': self.UNet_denoiser.state_dict(),
                    'unet_pvc': self.UNet_pvc.state_dict(),
                    'unet_denoiser_opt': self.unet_denoiser_optimizer.state_dict(),
                    'unet_pvc_opt': self.unet_pvc_optimizer.state_dict(),
                    'unet_denoiser_losses': self.unet_denoiser_list_losses,
                    'unet_pvc_losses': self.unet_pvc_list_losses,
                    'test_error': self.test_error,
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

        self.UNet_denoiser.load_state_dict(checkpoint['unet_denoiser'])
        self.UNet_pvc.load_state_dict(checkpoint['unet_pvc'])

        self.unet_denoiser_optimizer.load_state_dict(checkpoint['unet_denoiser_opt'])
        self.unet_pvc_optimizer.load_state_dict(checkpoint['unet_pvc_opt'])

        self.unet_denoiser_list_losses = checkpoint['unet_denoiser_losses']
        self.unet_pvc_list_losses = checkpoint['unet_pvc_losses']
        self.denoiser_loss = torch.Tensor([self.unet_denoiser_list_losses[-1]])
        self.pvc_loss = torch.Tensor([self.unet_pvc_list_losses])

        self.test_error = checkpoint['test_error']
        self.current_epoch = checkpoint['epoch']
        self.start_epoch=self.current_epoch

    def switch_eval(self):
        self.UNet_denoiser.eval()
        self.UNet_pvc.eval()

    def switch_train(self):
        self.UNet_denoiser.train()
        self.UNet_pvc.train()

    def switch_device(self, device):
        self.device = device
        self.UNet_denoiser.to(device=device)
        self.UNet_pvc.to(device=device)

    def show_infos(self, mse = False):
        formatted_params = self.format_params()
        print('*'*80)
        print('PARAMETRES (json param file) : \n')
        print(formatted_params)
        print('*' * 80)

        print('MEAN SQUARE ERROR on TEST DATA')
        print(f'list of MSE on test data :{self.test_error} ')
        print('*' * 80)
        print('DENOISER : ')
        print(self.UNet_denoiser)
        print('*' * 80)
        print('DEEPPVC : ')
        print(self.UNet_pvc)
        print('*'*80)

        # helpers_params.make_and_print_params_info_table([self.params])