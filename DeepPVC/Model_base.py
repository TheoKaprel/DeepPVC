import torch
import json
import copy
from abc import abstractmethod

from . import  helpers




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

        if "with_att" in params:
            self.with_att = params["with_att"]
        else:
            self.with_att = False


        if self.params["inputs"] == "projs":
            if params['sino']==False:
                self.input_channels = params['input_eq_angles']

                self.input_channels = self.input_channels+2*params['nb_adj_angles'] if params['with_adj_angles'] else self.input_channels
                self.input_channels = self.input_channels+1 if params['with_rec_fp'] else self.input_channels
            else:
                self.input_channels = params['input_eq_angles']+1
                self.input_channels = self.input_channels + 1 if params['with_rec_fp'] else self.input_channels


            self.output_channels_denoiser = self.input_channels - 1 if params['with_rec_fp'] else self.input_channels
            self.output_channels = 1

        elif self.params["inputs"]=="full_sino":
            self.input_channels = 256 if (params['sino']) else 120
            self.input_channels = 2 * self.input_channels if params['with_rec_fp'] else self.input_channels
            self.output_channels=self.output_channels_denoiser = self.input_channels // 2 if params['with_rec_fp'] else self.input_channels


        if ("dim" in self.params and self.params['dim']=="3d"):
            if self.with_att:
                self.input_channels = 3 if params['with_rec_fp'] else 2
            else:
                self.input_channels = 2 if params['with_rec_fp'] else 1

            if "with_PVCNet_rec" in self.params:
                if self.params["with_PVCNet_rec"]:
                    self.input_channels+=1

            self.output_channels = self.output_channels_denoiser = 1


        if "recon_loss" in params:
            self.with_lesion=("lesion" in params["recon_loss"])
            self.with_conv_loss = ("conv" in params['recon_loss'])
        else:
            self.with_lesion = ("lesion" in params["img_loss"]) or ("lesion" in params["sino_loss"])
            self.with_conv_loss = ("conv" in params["img_loss"]) or ("conv" in params["sino_loss"])

        self.use_dropout = params['use_dropout'] if 'use_dropout' in params else None
        self.leaky_relu = params['leaky_relu'] if 'leaky_relu' in params else None
        self.optimizer = params['optimizer']
        self.weight_decay = params['weight_decay'] if 'weight_decay' in params else 0

        self.output_folder = self.params['output_folder']
        self.output_pth = self.params['output_pth']

        self.resume_training = resume_training

        self.val_error_MSE,self.val_error_MAE=[],[]

        self.ones = torch.tensor([1.0],device=self.device,requires_grad=False)
        self.zeros = torch.tensor([0.0],device=self.device,requires_grad=False)

    @abstractmethod
    def input_data(self, batch_inputs, batch_targets):
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
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


