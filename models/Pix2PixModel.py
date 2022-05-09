from torch import nn,optim
from . import networks



class PVEPix2PixModel():
    def __init__(self, training_params):

        self.n_epochs = training_params['n_epochs']
        self.learning_rate = training_params['learning_rate']
        self.optimizer = training_params['optimizer']

        self.input_channels = training_params['input_channels']
        self.hidden_channels_gen = training_params['hidden_channels_gen']
        self.hidden_channels_disc = training_params['hidden_channels_disc']

        self.display_step = training_params['display_step']

        self.training_device = training_params['training_device']



        self.Generator = networks.UNetGenerator(input_channel=self.input_channels,
                                                ngc = self.hidden_channels_gen,
                                                output_channel=self.input_channels)

        self.Discriminator = networks.NLayerDiscriminator(input_channel=2*self.input_channels,
                                                          ndc = self.hidden_channels_disc,
                                                          output_channel=self.input_channels)

        if self.optimizer=='Adam':
            self.generator_optimizer =optim.Adam(self.Generator.parameters(), lr=self.learning_rate)
            self.discriminator_optimizer =optim.Adam(self.Discriminator.parameters(), lr=self.learning_rate)

