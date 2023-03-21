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
        self.input_channels = params['input_channels']

        self.use_dropout = params['use_dropout']
        self.leaky_relu = params['leaky_relu']
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
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


