import os.path
import time
import torch
from torch import optim
import json
import copy
from abc import abstractmethod

from . import networks, losses,plots, helpers
from torch.cuda.amp import autocast, GradScaler

class ModelInstance():
    def __new__(cls, params, from_pth = None, resume_training=False, device = None):
        network_architecture = params['network']

        if network_architecture == 'pix2pix':
            return Pix2PixModel(params=params, from_pth=from_pth,resume_training=resume_training, device=device)
        elif network_architecture == 'unet':
            return UNetModel(params=params, from_pth=from_pth,resume_training=resume_training, device=device)
        elif network_architecture == 'unet_denoiser_pvc':
            return UNet_Denoiser_PVC(params=params, from_pth=from_pth,resume_training=resume_training, device=device)
        elif network_architecture == 'gan_denoiser_pvc':
            return GAN_Denoiser_PVC(params=params, from_pth=from_pth,resume_training=resume_training, device=device)
        else:
            print(f"ERROR : unknown network architecture ({network_architecture})")
            exit(0)


class ModelBase(torch.nn.Module):
    def __init__(self,  params, resume_training, device = None):
        super(ModelBase, self).__init__()
        self.params = params
        if device==None:
            self.device = helpers.get_auto_device(self.params['device'])
        else:
            self.device = device

        self.n_epochs = params['n_epochs']
        self.learning_rate = params['learning_rate']
        self.input_channels = params['input_channels']

        self.use_dropout = params['use_dropout']
        self.leaky_relu = params['leaky_relu']
        self.sum_norm = params['sum_norm']
        self.optimizer = params['optimizer']

        self.output_folder = self.params['output_folder']
        self.output_pth = self.params['output_pth']

        self.resume_training = resume_training

        self.ones = torch.tensor([1.0],device=self.device,requires_grad=False)
        self.zeros = torch.tensor([0.0],device=self.device,requires_grad=False)

    @abstractmethod
    def input_data(self, batch):
        pass

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

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad



class Pix2PixModel(ModelBase):
    def __init__(self, params, from_pth = None,resume_training=False, device=None):
        assert (params['network'] == 'pix2pix')
        super().__init__(params,resume_training,device=device)

        self.conv3d = params['conv3d']
        self.init_feature_kernel = params['init_feature_kernel']
        self.nb_ed_layers = params['nb_ed_layers']
        self.hidden_channels_gen = params['hidden_channels_gen']
        self.hidden_channels_disc = params['hidden_channels_disc']
        self.generator_activation = params['generator_activation']
        self.layer_norm = params['layer_norm']
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

        self.scaler = GradScaler()




    def init_model(self):
        print('models device is supposed to be : ')
        print(self.device)
        self.Generator = networks.UNet(input_channel=self.input_channels, ngc = self.hidden_channels_gen,conv3d=self.conv3d,init_feature_kernel=self.init_feature_kernel, nb_ed_layers=self.nb_ed_layers,
                                                output_channel= 1 , generator_activation = self.generator_activation,use_dropout=self.use_dropout, leaky_relu = self.leaky_relu,
                                                sum_norm = self.sum_norm,norm = self.layer_norm, vmin=self.vmin).to(device=self.device)

        self.Discriminator = networks.NEncodingLayers(input_channel=self.input_channels+1,ndc = self.hidden_channels_disc,norm=self.layer_norm,
                                                    output_channel=1,leaky_relu=self.leaky_relu).to(device=self.device)

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
        self.scaler.scale(self.disc_loss).backward()
        self.scaler.step(self.discriminator_optimizer)

    def forward_G(self):
        self.GfakePVfree = self.Generator(self.truePVE)
        self.Gdisc_fake_hat = self.Discriminator(self.GfakePVfree, self.truePVE)

    def losses_G(self):
        self.gen_loss = self.losses.get_gen_loss(self.Gdisc_fake_hat, self.truePVfree, self.GfakePVfree)

    def backward_G(self):
        self.scaler.scale(self.gen_loss).backward()
        self.scaler.step(self.generator_optimizer)

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
            with autocast():
                self.forward_D()
                self.losses_D()
            self.backward_D()
            self.scaler.update()

        # Generator Updates
        self.set_requires_grad(self.Discriminator, requires_grad=False)
        for _ in range(self.generator_update):
            self.generator_optimizer.zero_grad()
            with autocast():
                self.forward_G()
                self.losses_G()
            self.backward_G()
            self.scaler.update()

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
        self.zeros = self.ones.to(device=device)
        self.losses.ones = self.losses.ones.to(device=device)
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

        # helpers_params.make_and_print_params_info_table([self.params])

class UNetModel(ModelBase):
    def __init__(self, params, from_pth = None,resume_training=False,device=None):
        assert (params['network'] == 'unet')
        super().__init__(params,resume_training,device=device)

        self.init_feature_kernel = params['init_feature_kernel']
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




    def init_model(self):
        self.UNet = networks.UNet(input_channel=self.input_channels, ngc = self.hidden_channels_unet,
                                  init_feature_kernel=self.init_feature_kernel,nb_ed_layers=self.nb_ed_layers,
                                  output_channel=1,generator_activation = self.unet_activation,use_dropout=self.use_dropout,
                                sum_norm = self.sum_norm,norm = self.unet_norm, vmin=self.vmin,leaky_relu = self.leaky_relu,).to(device=self.device)

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

        self.losses = losses.UNetLosses(self.losses_params)

    def input_data(self, batch):
        self.truePVE = batch[:, 0, :, :, :].to(self.device).float()
        self.truePVfree = batch[:, -1, 0:1, :, :].to(self.device).float()


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
        self.unet_losses = checkpoint['unet_losses']
        self.test_error = checkpoint['test_error']
        self.current_epoch = checkpoint['epoch']

        self.UNet.load_state_dict(checkpoint['unet'])

        if self.resume_training:
            self.unet_optimizer.load_state_dict(checkpoint['unet_opt'])
            self.init_optimization()
            self.init_losses()

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
    def __init__(self, params, from_pth = None,resume_training=False,device=None):
        assert(params['network']=='unet_denoiser_pvc')
        super().__init__(params,resume_training,device=device)

        self.init_feature_kernel_denoiser = params['init_feature_kernel_denoiser']
        self.nb_ed_layers_denoiser = params['nb_ed_layers_denoiser']
        self.hidden_channels_unet_denoiser = params['hidden_channels_unet_denoiser']
        self.unet_denoiser_activation = params['unet_denoiser_activation']
        self.unet_denoiser_norm = params['unet_denoiser_norm']

        self.init_feature_kernel_pvc = params['init_feature_kernel_pvc']
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


    def init_model(self):
        self.UNet_denoiser = networks.UNet(input_channel=self.input_channels, ngc = self.hidden_channels_unet_denoiser,
                                           init_feature_kernel=self.init_feature_kernel_denoiser,nb_ed_layers=self.nb_ed_layers_denoiser,
                                                output_channel=self.input_channels,generator_activation = self.unet_denoiser_activation,use_dropout=self.use_dropout,
                                                sum_norm = self.sum_norm,norm = self.unet_denoiser_norm, vmin=self.vmin,leaky_relu = self.leaky_relu).to(device=self.device)

        self.UNet_pvc = networks.UNet(input_channel=self.input_channels, ngc = self.hidden_channels_unet_pvc,
                                  init_feature_kernel=self.init_feature_kernel_pvc,nb_ed_layers=self.nb_ed_layers_pvc,
                                    output_channel=1,generator_activation = self.unet_pvc_activation,use_dropout=self.use_dropout,
                                    sum_norm = self.sum_norm,norm = self.unet_pvc_norm, vmin=self.vmin,leaky_relu = self.leaky_relu).to(device=self.device)

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
        self.denoiser_losses_params = {'recon_loss': self.params['recon_loss_denoiser'], 'lambda_losses': self.params['lambda_losses_denoiser']}
        self.pvc_losses_params = {'recon_loss': self.params['recon_loss_pvc'],  'lambda_losses': self.params['lambda_losses_pvc']}

        self.unet_denoiser_losses = losses.UNetLosses(self.denoiser_losses_params)
        self.unet_pvc_losses = losses.UNetLosses(self.pvc_losses_params)

    def input_data(self, batch):
        self.noisyPVE = batch[:, 0,:, :, :].to(self.device).float()
        self.truePVE = batch[:, 1,:, :, :].to(self.device).float()
        self.truePVfree = batch[:, 2,0:1, :, :].to(self.device).float()


    def forward_denoiser(self):
        self.fakePVE = self.UNet_denoiser(self.noisyPVE)

    def forward_pvc(self):
        with torch.no_grad():
            self.fakePVE = self.UNet_denoiser(self.noisyPVE)
        self.fakePVfree = self.UNet_pvc(self.fakePVE.detach())

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

        self.denoisedPVE = self.UNet_denoiser(self.noisyPVE)
        return self.UNet_pvc(self.denoisedPVE)


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
                    }, output_path)
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
        self.UNet_denoiser.load_state_dict(checkpoint['unet_denoiser'])
        self.UNet_pvc.load_state_dict(checkpoint['unet_pvc'])

        self.test_error = checkpoint['test_error']
        self.current_epoch = checkpoint['epoch']

        self.unet_denoiser_list_losses = checkpoint['unet_denoiser_losses']
        self.unet_pvc_list_losses = checkpoint['unet_pvc_losses']

        if self.resume_training:
            self.init_optimization()
            self.init_losses()
            self.unet_denoiser_optimizer.load_state_dict(checkpoint['unet_denoiser_opt'])
            self.unet_pvc_optimizer.load_state_dict(checkpoint['unet_pvc_opt'])


            self.denoiser_loss = torch.Tensor([self.unet_denoiser_list_losses[-1]])
            self.pvc_loss = torch.Tensor([self.unet_pvc_list_losses])


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



class GAN_Denoiser_PVC(ModelBase):
    def __init__(self, params, from_pth = None,resume_training=False,device=None):
        assert(params['network']=='gan_denoiser_pvc')
        super().__init__(params,resume_training,device=device)

        self.init_feature_kernel_denoiser = params['init_feature_kernel_denoiser']
        self.nb_ed_layers_gen_denoiser = params['nb_ed_layers_gen_denoiser']
        self.hidden_channels_gen_denoiser = params['hidden_channels_gen_denoiser']
        self.gen_denoiser_norm = params['gen_denoiser_norm']
        self.gen_denoiser_activation = params['gen_denoiser_activation']
        self.hidden_channels_disc_denoiser = params['hidden_channels_disc_denoiser']
        self.recon_loss_denoiser = params['recon_loss_denoiser']
        self.adv_loss_denoiser = params['adv_loss_denoiser']
        self.lambda_recon_denoiser = params['lambda_recon_denoiser']
        self.generator_update_denoiser = params['generator_update_denoiser']
        self.discriminator_update_denoiser = params['discriminator_update_denoiser']

        self.init_feature_kernel_pvc = params['init_feature_kernel_pvc']
        self.nb_ed_layers_gen_pvc = params['nb_ed_layers_gen_pvc']
        self.hidden_channels_gen_pvc = params['hidden_channels_gen_pvc']
        self.gen_pvc_norm = params['gen_pvc_norm']
        self.gen_pvc_activation = params['gen_pvc_activation']
        self.hidden_channels_disc_pvc = params['hidden_channels_disc_pvc']
        self.recon_loss_pvc = params['recon_loss_pvc']
        self.adv_loss_pvc = params['adv_loss_pvc']
        self.lambda_recon_pvc = params['lambda_recon_pvc']
        self.generator_update_pvc = params['generator_update_pvc']
        self.discriminator_update_pvc = params['discriminator_update_pvc']

        self.denoiser_update = params['denoiser_update']
        self.pvc_update = params['pvc_update']

        if (self.gen_denoiser_activation=='relu_min' or self.gen_pvc_activation=='relu_min'):
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

            self.gen_denoiser_list_losses = []
            self.disc_denoiser_list_losses = []
            self.gen_pvc_list_losses = []
            self.disc_pvc_list_losses = []

            self.test_error = []

            self.current_epoch = 1
            self.start_epoch=1

        self.current_iteration = 0
        self.mean_gen_denoiser_losses = 0
        self.mean_disc_denoiser_losses = 0
        self.mean_gen_pvc_losses = 0
        self.mean_disc_pvc_losses = 0


    def init_model(self):
        # --------DENOISER-----
        self.Denoiser_Generator = networks.UNet(input_channel=self.input_channels, ngc=self.hidden_channels_gen_denoiser,
                                                init_feature_kernel=self.init_feature_kernel_denoiser,nb_ed_layers=self.nb_ed_layers_gen_denoiser,
                                                output_channel= self.input_channels, generator_activation=self.gen_denoiser_activation, use_dropout=self.use_dropout,
                                                sum_norm=self.sum_norm, norm=self.gen_denoiser_norm, vmin=self.vmin,leaky_relu = self.leaky_relu).to(device=self.device)

        self.Denoiser_Discriminator = networks.NEncodingLayers(input_channel=2*self.input_channels,
                                                               ndc=self.hidden_channels_disc_denoiser,
                                                               output_channel=self.input_channels,leaky_relu=self.leaky_relu).to(device=self.device)
        #---------PVC----------
        self.PVC_Generator = networks.UNet(input_channel=self.input_channels,ngc=self.hidden_channels_gen_pvc,
                                           init_feature_kernel=self.init_feature_kernel_pvc,nb_ed_layers=self.nb_ed_layers_gen_pvc,
                                            output_channel=1,generator_activation=self.gen_pvc_activation,use_dropout=self.use_dropout,
                                            sum_norm=self.sum_norm, norm=self.gen_pvc_norm, vmin=self.vmin,leaky_relu = self.leaky_relu).to(device=self.device)

        self.PVC_Discriminator = networks.NEncodingLayers(input_channel= self.input_channels + 1,
                                                            ndc=self.hidden_channels_disc_pvc,
                                                            output_channel=1,leaky_relu=self.leaky_relu).to(device=self.device)


    def init_optimization(self):
        if self.optimizer == 'Adam':
            self.denoiser_gen_optimizer = optim.Adam(self.Denoiser_Generator.parameters(), lr=self.learning_rate)
            self.denoiser_disc_optimizer = optim.Adam(self.Denoiser_Discriminator.parameters(), lr=self.learning_rate)
            self.pvc_gen_optimizer = optim.Adam(self.PVC_Generator.parameters(), lr=self.learning_rate)
            self.pvc_disc_optimizer = optim.Adam(self.PVC_Discriminator.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Unknown optimizer. Choose between : Adam")

        self.learning_rate_policy_infos = self.params['lr_policy']
        if self.learning_rate_policy_infos[0]=='multiplicative':
            mult_rate = self.learning_rate_policy_infos[1]
            lbda = lambda epoch: mult_rate

            self.scheduler_denoiser_gen = optim.lr_scheduler.MultiplicativeLR(self.denoiser_gen_optimizer, lbda)
            self.scheduler_denoiser_disc = optim.lr_scheduler.MultiplicativeLR(self.denoiser_disc_optimizer, lbda)
            self.scheduler_pvc_gen = optim.lr_scheduler.MultiplicativeLR(self.pvc_gen_optimizer, lbda)
            self.scheduler_pvc_disc = optim.lr_scheduler.MultiplicativeLR(self.pvc_disc_optimizer, lbda)



    def init_losses(self):
        self.denoiser_gen_loss = torch.Tensor([0])
        self.denoiser_disc_loss = torch.Tensor([0])
        self.pvc_gen_loss = torch.Tensor([0])
        self.pvc_disc_loss = torch.Tensor([0])

        self.gp_denoiser = self.params['with_gradient_penalty_denoiser']
        self.gp_pvc = self.params['with_gradient_penalty_pvc']

        self.denoiser_losses_params = {'adv_loss': self.adv_loss_denoiser, 'recon_loss': self.recon_loss_denoiser, 'lambda_recon': self.lambda_recon_denoiser,'gradient_penalty': self.gp_denoiser, 'device':self.device}
        self.pvc_losses_params = {'adv_loss': self.adv_loss_pvc, 'recon_loss': self.recon_loss_pvc, 'lambda_recon': self.lambda_recon_pvc, 'gradient_penalty': self.gp_pvc,'device':self.device}

        self.denoiser_losses = losses.Pix2PixLosses(self.denoiser_losses_params)
        self.pvc_losses = losses.Pix2PixLosses(self.pvc_losses_params)

    def input_data(self, batch):
        self.noisyPVE = batch[:, 0,:, :, :].to(self.device).float()
        self.truePVE = batch[:, 1,:, :, :].to(self.device).float()
        self.truePVfree = batch[:, 2,0:1, :, :].to(self.device).float()



    def forward_denoiser_G(self):
        self.fakePVE = self.Denoiser_Generator(self.noisyPVE)
        self.disc_fakePVE_hat = self.Denoiser_Discriminator(self.fakePVE, self.noisyPVE)

    def backward_denoiser_G(self):
        self.denoiser_gen_loss = self.denoiser_losses.get_gen_loss(self.disc_fakePVE_hat, self.truePVE, self.fakePVE)
        self.denoiser_gen_loss.backward()
        self.denoiser_gen_optimizer.step()

    def forward_denoiser_D(self):
        with torch.no_grad():
            self.DfakePVE = self.Denoiser_Generator(self.noisyPVE)

        self.Ddisc_fakePVE_hat = self.Denoiser_Discriminator(self.DfakePVE.detach(), self.noisyPVE)
        self.Ddisc_truePVE_hat = self.Denoiser_Discriminator(self.truePVE, self.noisyPVE)

    def backward_denoiser_D(self):
        disc_fakePVE_loss = self.denoiser_losses.adv_loss(self.Ddisc_fakePVE_hat, torch.zeros_like(self.Ddisc_fakePVE_hat))
        disc_truePVE_loss = self.denoiser_losses.adv_loss(self.Ddisc_truePVE_hat, torch.ones_like(self.Ddisc_fakePVE_hat))
        self.denoiser_disc_loss = disc_fakePVE_loss+disc_truePVE_loss

        if self.gp_denoiser:
            self.denoiser_disc_loss += 10 * self.denoiser_losses.get_gradient_penalty(Discriminator=self.Denoiser_Discriminator, real = self.truePVE, fake = self.DfakePVE, condition=self.noisyPVE)

        self.denoiser_disc_loss.backward(retain_graph=True)
        self.denoiser_disc_optimizer.step()


    def forward_pvc_G(self):
        with torch.no_grad():
            self.fakePVE = self.Denoiser_Generator(self.noisyPVE)
        self.fakePVfree = self.PVC_Generator(self.fakePVE.detach())
        self.disc_fakePVfree_hat = self.PVC_Discriminator(self.fakePVfree, self.fakePVE.detach())

    def backward_pvc_G(self):
        self.pvc_gen_loss = self.pvc_losses.get_gen_loss(self.disc_fakePVfree_hat, self.truePVfree, self.fakePVfree)
        self.pvc_gen_loss.backward()
        self.pvc_gen_optimizer.step()

    def forward_pvc_D(self):
        with torch.no_grad():
            self.fakePVE = self.Denoiser_Generator(self.noisyPVE)
            self.DfakePVfree = self.PVC_Generator(self.fakePVE.detach())

        self.Ddisc_fakePVfree_hat = self.PVC_Discriminator(self.DfakePVfree.detach(), self.fakePVE.detach())
        self.Ddisc_truePVfree_hat = self.PVC_Discriminator(self.truePVfree,self.fakePVE.detach())

    def backward_pvc_D(self):
        disc_fakePVfree_loss = self.pvc_losses.adv_loss(self.Ddisc_fakePVfree_hat, torch.zeros_like(self.Ddisc_fakePVfree_hat))
        disc_truePVfree_loss = self.pvc_losses.adv_loss(self.Ddisc_truePVfree_hat, torch.ones_like(self.Ddisc_truePVfree_hat))
        self.pvc_disc_loss = disc_fakePVfree_loss+disc_truePVfree_loss

        if self.gp_pvc:
            self.pvc_disc_loss += 10 * self.pvc_losses.get_gradient_penalty(Discriminator=self.PVC_Discriminator, real = self.truePVfree, fake = self.DfakePVfree, condition=self.fakePVE.detach())


        self.pvc_disc_loss.backward(retain_graph=True)
        self.pvc_disc_optimizer.step()

    def forward(self, batch):
        if batch.dim()==4:
            self.noisyPVE = batch.to(self.device).float()
        elif batch.dim()==5:
            self.noisyPVE = batch[:,0,:,:,:].to(self.device).float()

        self.denoisedPVE = self.Denoiser_Generator(self.noisyPVE)
        self.fakePVfree = self.PVC_Generator(self.denoisedPVE)
        return self.fakePVfree

    def optimize_parameters(self):
        # denoiser updates
        for _denoiser in range(self.denoiser_update):

            # denoiser discriminator updates
            for _denoiser_disc in range(self.discriminator_update_denoiser):
                self.denoiser_disc_optimizer.zero_grad()
                self.forward_denoiser_D()
                self.backward_denoiser_D()
            self.mean_disc_denoiser_losses += self.denoiser_disc_loss.item()

            # denoiser generator updates
            for _denoiser_gen in range(self.generator_update_denoiser):
                self.denoiser_gen_optimizer.zero_grad()
                self.forward_denoiser_G()
                self.backward_denoiser_G()
            self.mean_gen_denoiser_losses += self.denoiser_gen_loss.item()

        # pvc update
        for _pvc in range(self.pvc_update):

            # pvc discriminator updates
            for _pvc_disc in range(self.discriminator_update_pvc):
                self.pvc_disc_optimizer.zero_grad()
                self.forward_pvc_D()
                self.backward_pvc_D()
            self.mean_disc_pvc_losses+=self.pvc_disc_loss.item()

            # pvc generator updates
            for _pvc_gen in range(self.generator_update_pvc):
                self.pvc_gen_optimizer.zero_grad()
                self.forward_pvc_G()
                self.backward_pvc_G()
            self.mean_gen_pvc_losses+=self.pvc_gen_loss.item()

        self.current_iteration+=1

    def update_epoch(self):
        self.disc_denoiser_list_losses.append(self.mean_disc_denoiser_losses / self.current_iteration)
        self.gen_denoiser_list_losses.append(self.mean_gen_denoiser_losses / self.current_iteration)
        self.disc_pvc_list_losses.append(self.mean_disc_pvc_losses / self.current_iteration)
        self.gen_pvc_list_losses.append(self.mean_gen_pvc_losses / self.current_iteration)

        self.current_epoch+=1
        self.current_iteration=0

        self.mean_disc_denoiser_losses, self.mean_gen_denoiser_losses = 0,0
        self.mean_disc_pvc_losses, self.mean_gen_pvc_losses = 0,0

        self.scheduler_denoiser_disc.step()
        self.scheduler_denoiser_gen.step()
        self.scheduler_pvc_disc.step()
        self.scheduler_pvc_gen.step()

    def plot_losses(self, save, wait, title):
        plots.plot_losses_double_double_model(losses1=self.gen_denoiser_list_losses, losses2=self.disc_denoiser_list_losses, losses3=self.gen_pvc_list_losses, losses4=self.disc_pvc_list_losses,
                                              test_mse=self.test_error, labels=['Denoiser Generator', 'Denoiser Discriminator', 'PVC Generator', 'PVC Discriminator'],
                                              save=save,wait=wait, title=title
                                              )
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

                    'denoiser_gen': self.Denoiser_Generator.state_dict(),
                    'denoiser_disc': self.Denoiser_Discriminator.state_dict(),
                    'pvc_gen': self.PVC_Generator.state_dict(),
                    'pvc_disc': self.PVC_Discriminator.state_dict(),

                    'denoiser_gen_opt': self.denoiser_gen_optimizer.state_dict(),
                    'denoiser_disc_opt': self.denoiser_disc_optimizer.state_dict(),
                    'pvc_gen_opt': self.pvc_gen_optimizer.state_dict(),
                    'pvc_disc_opt': self.pvc_disc_optimizer.state_dict(),

                    'denoiser_gen_losses': self.gen_denoiser_list_losses,
                    'denoiser_disc_losses': self.disc_denoiser_list_losses,
                    'pvc_gen_losses': self.gen_pvc_list_losses,
                    'pvc_disc_losses': self.disc_pvc_list_losses,

                    'test_error': self.test_error,
                    'params': self.params
                    }, output_path)
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

        self.Denoiser_Generator.load_state_dict(checkpoint['denoiser_gen'])
        self.Denoiser_Discriminator.load_state_dict(checkpoint['denoiser_disc'])
        self.PVC_Generator.load_state_dict(checkpoint['pvc_gen'])
        self.PVC_Discriminator.load_state_dict(checkpoint['pvc_disc'])

        self.gen_denoiser_list_losses = checkpoint['denoiser_gen_losses']
        self.disc_denoiser_list_losses = checkpoint['denoiser_disc_losses']
        self.gen_pvc_list_losses = checkpoint['pvc_gen_losses']
        self.disc_pvc_list_losses = checkpoint['pvc_disc_losses']

        self.test_error = checkpoint['test_error']
        self.current_epoch = checkpoint['epoch']


        if self.resume_training:
            self.denoiser_gen_optimizer.load_state_dict(checkpoint['denoiser_gen_opt'])
            self.denoiser_disc_optimizer.load_state_dict(checkpoint['denoiser_disc_opt'])
            self.pvc_gen_optimizer.load_state_dict(checkpoint['pvc_gen_opt'])
            self.pvc_disc_optimizer.load_state_dict(checkpoint['pvc_disc_opt'])

            self.denoiser_gen_loss = torch.Tensor([self.gen_denoiser_list_losses[-1]])
            self.denoiser_disc_loss = torch.Tensor([self.disc_denoiser_list_losses[-1]])
            self.pvc_gen_loss = torch.Tensor([self.gen_pvc_list_losses[-1]])
            self.pvc_disc_loss = torch.Tensor([self.disc_pvc_list_losses[-1]])

            self.start_epoch=self.current_epoch

    def switch_eval(self):
        self.Denoiser_Generator.eval()
        self.Denoiser_Discriminator.eval()
        self.PVC_Generator.eval()
        self.PVC_Discriminator.eval()

    def switch_train(self):
        self.Denoiser_Generator.train()
        self.Denoiser_Discriminator.train()
        self.PVC_Generator.train()
        self.PVC_Discriminator.train()

    def switch_device(self, device):
        self.device = device
        self.Denoiser_Generator.to(device=device)
        self.Denoiser_Discriminator.to(device=device)
        self.PVC_Generator.to(device=device)
        self.PVC_Discriminator.to(device=device)

    def show_infos(self, mse = False):
        formatted_params = self.format_params()
        print('*'*80)
        print('PARAMETRES (json param file) : \n')
        print(formatted_params)
        print('*' * 80)

        print('MEAN ERROR on TEST DATA')
        print(f'list of ERRORS on test data :{self.test_error} ')
        print('*' * 80)
        print('DENOISER : ')
        print('     GENERATOR : ')
        print(self.Denoiser_Generator)
        print('     DISCRIMINATOR : ')
        print(self.Denoiser_Discriminator)
        print('*' * 80)
        print('DEEPPVC : ')
        print('     GENERATOR : ')
        print(self.PVC_Generator)
        print('     DISCRIMINATOR : ')
        print(self.PVC_Discriminator)
        print('*'*80)