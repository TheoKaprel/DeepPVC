import os.path
import time
import torch
from torch import optim

from . import networks, losses,helpers_data_parallelism,networks_diff
from torch.cuda.amp import autocast, GradScaler

from .Model_base import ModelBase

class UNetModel(ModelBase):
    def __init__(self, params, from_pth = None,resume_training=False, device=None):
        assert (params['network'] == 'unet')
        super().__init__(params,resume_training,device=device)

        self.verbose=params['verbose']
        self.conv3d = params['conv3d']
        self.init_feature_kernel = params['init_feature_kernel']
        self.nb_ed_layers = params['nb_ed_layers']
        if "ed_blocks" in params:
            self.ed_blocks = params["ed_blocks"]
        else:
            self.ed_blocks = "conv-relu-norm"

        self.hidden_channels_unet = params["hidden_channels_unet"]
        self.unet_activation = params["unet_activation"]

        self.layer_norm = params['layer_norm']
        self.residual_layer=params['residual_layer']
        self.attention=False if 'attention' not in params else params['attention']

        self.init_model()

        if from_pth:
            if self.verbose>1:
                print('normalement self.load_model(from_pth) mais là non, on le fait juste apres l initialisation des gpus etc')
        else:

            self.init_optimization()
            self.init_losses()

            self.unet_losses = []
            self.test_error = []

            self.current_epoch = 1
            self.start_epoch=1

        self.current_iteration = 0
        self.mean_unet_loss = 0

        self.amp = self.params['amp']
        if self.amp:
            self.scaler = GradScaler()
        self.autocat_losses= self.amp


    def init_model(self):
        if self.verbose>0:
            print(f'models device is supposed to be : {self.device}')
        if self.attention:
            self.UNet = networks_diff.AttentionResUnet(init_dim=self.hidden_channels_unet,out_dim=1,channels=self.input_channels,dim_mults=(1,2,4,8)).to(device = self.device)
        else:
            self.UNet = networks.UNet(input_channel=self.input_channels, ngc = self.hidden_channels_unet,conv3d=self.conv3d,init_feature_kernel=self.init_feature_kernel, nb_ed_layers=self.nb_ed_layers,
                                                output_channel= 1, generator_activation = self.unet_activation,use_dropout=self.use_dropout, leaky_relu = self.leaky_relu,
                                                norm = self.layer_norm, residual_layer=self.residual_layer, blocks = self.ed_blocks).to(device=self.device)

        if self.params['jean_zay']:
            helpers_data_parallelism.init_data_parallelism(model=self)

    def init_optimization(self):
        if self.optimizer == 'Adam':
            self.unet_optimizer = optim.Adam(self.UNet.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Unknown optimizer. Choose between : Adam")

        self.learning_rate_policy_infos = self.params['lr_policy']
        if self.learning_rate_policy_infos[0]=='multiplicative':
            mult_rate = self.learning_rate_policy_infos[1]
            self.update_lr_every = self.learning_rate_policy_infos[2]
            lbda = lambda epoch: mult_rate

            self.scheduler_unet = optim.lr_scheduler.MultiplicativeLR(self.unet_optimizer, lbda)

    def init_losses(self):
        self.losses_params = {'recon_loss': self.params['recon_loss'],
                              'lambda_recon': self.params['lambda_recon'],'device':self.device}
        self.losses = losses.UNetLosses(self.losses_params)


    def input_data(self, batch):
        self.truePVE = batch[:, 0, :, :, :]
        self.truePVfree = batch[:, -1, 0:1, :, :]


    def forward_unet(self):
        self.fakePVfree = self.UNet(self.truePVE)

    def losses_unet(self):
        self.unet_loss = self.losses.get_unet_loss(target=self.truePVfree, output=self.fakePVfree)

    def backward_unet(self):
        if self.amp:
            self.scaler.scale(self.unet_loss).backward()
            self.scaler.step(self.unet_optimizer)
        else:
            self.unet_loss.backward()
            self.unet_optimizer.step()

    def forward(self, batch):
        if  batch.dim()==4:
            self.truePVE = batch
        elif batch.dim()==5:
            self.truePVE = batch[:, 0,:, :, :]

        return self.UNet(self.truePVE)


    def optimize_parameters(self):
        # Unet Update
        self.set_requires_grad(self.UNet, requires_grad=True)
        self.unet_optimizer.zero_grad()
        with autocast(enabled=self.amp):
            self.forward_unet()
            self.losses_unet()
            self.backward_unet()
            if self.amp:
                self.scaler.update()

        self.mean_unet_loss+=self.unet_loss.item()
        self.current_iteration+=1

    def update_epoch(self):
        self.unet_losses.append(self.mean_unet_loss / self.current_iteration)
        if self.verbose > 1:
            print(f'Unet loss : {round(self.unet_losses[-1],5)}')


        if self.current_epoch % self.update_lr_every ==0:
            self.scheduler_unet.step()

        if self.verbose > 1:
            print(f'next lr (G): {self.scheduler_unet.get_last_lr()}')

        self.current_epoch+=1
        self.current_iteration=0
        self.mean_unet_loss = 0

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
                        'unet': self.UNet.module.state_dict(),
                        'unet_opt': self.unet_optimizer.state_dict(),
                        'unet_losses': self.unet_losses,
                        'test_error': self.test_error,
                        'params': self.params
                        }, output_path )
        else:
            torch.save({'saving_date': time.asctime(),
                        'epoch': self.current_epoch,
                        'unet': self.UNet.state_dict(),
                        'unet_opt': self.unet_optimizer.state_dict(),
                        'unet_losses': self.unet_losses,
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

        if hasattr(self.UNet, 'module'):
            self.UNet.module.load_state_dict(checkpoint['unet'])
        else:
            self.UNet.load_state_dict(checkpoint['unet'])
        self.unet_losses = checkpoint['unet_losses']
        self.test_error = checkpoint['test_error']
        self.current_epoch = checkpoint['epoch']

        if self.resume_training:
            self.init_optimization()
            self.init_losses()
            self.unet_optimizer.load_state_dict(checkpoint['unet_opt'])
            self.start_epoch=self.current_epoch

    def switch_eval(self):
        self.UNet.eval()

    def switch_train(self):
        self.UNet.train()

    def switch_device(self, device):
        self.device = device
        self.UNet = self.UNet.to(device=device)

    def show_infos(self, mse = False):
        formatted_params = self.format_params()
        print('*'*80)
        print('PARAMETRES (json param file) : \n')
        print(formatted_params)
        print('*' * 80)

        print('MEAN SQUARE ERROR on TEST DATA')
        print(f'list of errors on test data :{self.test_error} ')
        print('*' * 80)
        print(self.UNet)
        print('*' * 80)
        if hasattr(self, "losses"):
            print('Losses : ')
            print(self.losses_params)
            print(self.losses)
            print('*' * 80)

        # helpers_params.make_and_print_params_info_table([self.params])

    def plot_losses(self, save, wait, title):
        print('not implemented sorry.')