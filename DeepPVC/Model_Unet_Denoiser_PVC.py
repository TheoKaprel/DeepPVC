import os.path
import time

import matplotlib.pyplot as plt
import torch
from torch import optim

from . import networks, losses, helpers_data_parallelism, networks_diff, plots,networks_attention
from torch.cuda.amp import autocast, GradScaler

from .Model_base import ModelBase
from torchscan import summary

class UNet_Denoiser_PVC(ModelBase):
    def __init__(self, params, from_pth=None, resume_training=False, device=None):
        assert (params['network'] == 'unet_denoiser_pvc')
        super().__init__(params, resume_training, device=device)
        self.network_type = 'unet_denoiser_pvc'
        self.verbose = params['verbose']

        if "dim" in params:
            if params["dim"]=="2d":
                self.dim=2
            elif params["dim"]=="3d":
                self.dim=3
        else:
            self.dim=2

        self.init_feature_kernel = params['init_feature_kernel']
        self.final_feature_kernel = params['final_feature_kernel'] if "final_feature_kernel" in params else 3
        self.kernel_size = params['kernel_size'] if "kernel_size" in params else 3
        self.nb_ed_layers = params['nb_ed_layers']
        if "ed_blocks" in params:
            self.ed_blocks = params["ed_blocks"]
        else:
            self.ed_blocks = "conv-relu-norm"

        self.with_rec_fp = params['with_rec_fp']

        self.hidden_channels_unet = params["hidden_channels_unet"]
        self.unet_activation = params["unet_activation"]

        self.layer_norm = params['layer_norm']
        self.residual_layer = params['residual_layer']
        self.ResUnet = params['resunet']
        self.attention = params['attention']

        if self.residual_layer:
            self.residual_channel = 1 if self.with_rec_fp else 0
        else:
            self.residual_channel = -1

        self.final_2dconv = True if (params['dim']=="3d" and params['inputs']=="projs") else False

        if "archi" not in params:
            self.archi = "unet"
        else:
            self.archi = params["archi"]

        self.denoise = params['denoise'] if "denoise" in params else True

        self.paths= params['paths'] if "paths" in params else False

        if from_pth:
            self.for_training=False
            self.init_model()
            if self.verbose > 1:
                print(
                    'normalement self.load_model(from_pth) mais là non, on le fait juste apres l initialisation des gpus etc')
        else:
            self.for_training = True
            self.init_model()
            self.init_optimization()
            self.init_losses()

            self.unet_denoiser_losses,self.unet_pvc_losses = [],[]
            self.test_error = []

            self.current_epoch = 1
            self.start_epoch = 1

        self.current_iteration = 0
            # self.mean_unet_denoiser_loss, self.mean_unet_pvc_loss = torch.tensor([0], device=self.device, dtype=torch.float64), torch.tensor([0],  device=self.device,  dtype=torch.float64)
        self.mean_unet_denoiser_loss, self.mean_unet_pvc_loss = 0,0

        # self.iter_loss=[]

        self.amp = self.params['amp']
        if self.amp:
            self.scaler = GradScaler()
        self.autocat_losses = self.amp

    def init_model(self):
        if self.verbose > 0:
            print(f'models device is supposed to be : {self.device}')
        if self.archi=="attention":
            # self.UNet = networks_diff.AttentionResUnet(init_dim=self.hidden_channels_unet, out_dim=1,
            #                                            channels=self.input_channels, dim_mults=(1, 2, 4, 8)).to(
            #     device=self.device)
            self.UNet_denoiser = networks_attention.R2AttU_Net(img_ch=self.input_channels,output_ch=self.input_channels,t=2).to(device=self.device)
            self.UNet_pvc = networks_attention.R2AttU_Net(img_ch=self.input_channels,output_ch=1,t=2).to(device=self.device)
        elif self.archi=="dcnn":
            # self.UNet_denoiser = networks.ResCNN(in_channels=self.input_channels,out_channels=self.input_channels,ngc=self.hidden_channels_unet).to(device=self.device)
            # self.UNet_pvc = networks.ResCNN(in_channels=self.input_channels, out_channels=1,ngc=self.hidden_channels_unet).to(device=self.device)
            self.UNet_denoiser = networks.vanillaCNN(input_channel=self.input_channels, ngc=self.hidden_channels_unet,
                                    dim=self.dim,init_feature_kernel=self.init_feature_kernel,
                                      nb_ed_layers=self.nb_ed_layers,
                                      output_channel=self.output_channels_denoiser, generator_activation=self.unet_activation,
                                      use_dropout=self.use_dropout, leaky_relu=self.leaky_relu,
                                      norm=self.layer_norm, residual_layer=self.residual_layer
                                      ).to(device=self.device)
            self.UNet_pvc = networks.vanillaCNN(input_channel=self.input_channels, ngc=self.hidden_channels_unet,
                                      dim=self.dim,init_feature_kernel=self.init_feature_kernel,
                                      nb_ed_layers=self.nb_ed_layers,
                                      output_channel=self.output_channels, generator_activation=self.unet_activation,
                                      use_dropout=self.use_dropout, leaky_relu=self.leaky_relu,
                                      norm=self.layer_norm, residual_layer=self.residual_layer
                                      ).to(device=self.device)
        elif self.archi=="unet":
            self.UNet_denoiser = networks.UNet(input_channel=self.input_channels, ngc=self.hidden_channels_unet,paths=self.paths,
                                    dim=self.dim,init_feature_kernel=self.init_feature_kernel,final_feature_kernel=self.final_feature_kernel,
                                      nb_ed_layers=self.nb_ed_layers,
                                      output_channel=self.output_channels_denoiser, generator_activation=self.unet_activation,
                                      use_dropout=self.use_dropout, leaky_relu=self.leaky_relu,
                                      norm=self.layer_norm, residual_layer=self.residual_channel, blocks=self.ed_blocks,
                                      ResUnet=self.ResUnet,
                                               final_2dconv=False).to(device=self.device)
            self.UNet_pvc = networks.UNet(input_channel=self.input_channels, ngc=self.hidden_channels_unet,paths=self.paths,final_feature_kernel=self.final_feature_kernel,
                                      dim=self.dim,init_feature_kernel=self.init_feature_kernel,
                                      nb_ed_layers=self.nb_ed_layers,
                                      output_channel=self.output_channels, generator_activation=self.unet_activation,
                                      use_dropout=self.use_dropout, leaky_relu=self.leaky_relu,
                                      norm=self.layer_norm, residual_layer=self.residual_channel, blocks=self.ed_blocks,
                                      ResUnet=self.ResUnet,
                                      final_2dconv=self.final_2dconv, final_2dchannels=2*self.params['nb_adj_angles'] if self.final_2dconv else 0).to(device=self.device)
        elif self.archi=="unet_sym":
            self.UNet_denoiser = networks.UNet_symetric(input_channel=self.input_channels, ngc=self.hidden_channels_unet,paths=self.paths,
                                    dim=self.dim,init_feature_kernel=self.init_feature_kernel,final_feature_kernel=self.final_feature_kernel,
                                      nb_ed_layers=self.nb_ed_layers,
                                      output_channel=self.output_channels_denoiser, generator_activation=self.unet_activation,
                                      use_dropout=self.use_dropout, leaky_relu=self.leaky_relu,
                                      norm=self.layer_norm, residual_layer=self.residual_channel, blocks=self.ed_blocks,
                                      ResUnet=self.ResUnet,attention=self.attention,
                                               final_2dconv=False,
                                                        kernel_size = self.kernel_size).to(device=self.device)
            self.UNet_pvc = networks.UNet_symetric(input_channel=self.input_channels, ngc=self.hidden_channels_unet,paths=self.paths,final_feature_kernel=self.final_feature_kernel,
                                      dim=self.dim,init_feature_kernel=self.init_feature_kernel,
                                      nb_ed_layers=self.nb_ed_layers,
                                      output_channel=self.output_channels, generator_activation=self.unet_activation,
                                      use_dropout=self.use_dropout, leaky_relu=self.leaky_relu,
                                      norm=self.layer_norm, residual_layer=self.residual_channel, blocks=self.ed_blocks,
                                      ResUnet=self.ResUnet,attention=self.attention,
                                      final_2dconv=self.final_2dconv, final_2dchannels=2*self.params['nb_adj_angles'] if self.final_2dconv else 0,
                                                   kernel_size = self.kernel_size).to(device=self.device)
        elif self.archi=="big3dunet":
            self.UNet_denoiser = networks.Big3DUnet(params=self.params, input_channels=self.input_channels).to(self.device)
            self.UNet_pvc = networks.Big3DUnet(params=self.params, input_channels=self.input_channels).to(self.device)

        if "init" not in self.params:
            self.params["init"] = "none"

        if self.for_training:
            if "init" not in self.params:
                self.params["init"] = "none"
            self.UNet_denoiser = networks.init_weights(net=self.UNet_denoiser, init_type=self.params["init"])
            self.UNet_pvc = networks.init_weights(net=self.UNet_pvc, init_type=self.params["init"])

        if self.params['jean_zay']:
            helpers_data_parallelism.init_data_parallelism(model=self)

        if (('compile' in self.params) and (self.params['compile'] == True)):
            self.compile = True
        else:
            self.compile = False

        if (self.compile and self.for_training):
            self.UNet_denoiser = torch.compile(self.UNet_denoiser)
            self.UNet_pvc = torch.compile(self.UNet_pvc)

    def init_optimization(self):
        if self.optimizer == 'Adam':
            self.double_optimizer = optim.Adam(list(self.UNet_denoiser.parameters())+list(self.UNet_pvc.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer =="AdamW":
            # self.unet_denoiser_optimizer = optim.AdamW(self.UNet_denoiser.parameters(), lr=self.learning_rate)
            # self.unet_pvc_optimizer = optim.AdamW(self.UNet_pvc.parameters(), lr=self.learning_rate)
            self.double_optimizer = optim.AdamW(list(self.UNet_denoiser.parameters())+list(self.UNet_pvc.parameters()), lr=self.learning_rate)
        elif self.optimizer =="SGD":
            # self.unet_denoiser_optimizer = optim.SGD(self.UNet_denoiser.parameters(), lr=self.learning_rate,momentum=0.9)
            # self.unet_pvc_optimizer = optim.SGD(self.UNet_pvc.parameters(), lr=self.learning_rate,momentum=0.9)
            self.double_optimizer = optim.SGD(list(self.UNet_denoiser.parameters())+list(self.UNet_pvc.parameters()), lr=self.learning_rate,momentum=0.9)
        elif self.optimizer=="RMSprop":
            # self.unet_denoiser_optimizer = optim.RMSprop(self.UNet_denoiser.parameters(), lr=self.learning_rate,momentum=0.9)
            # self.unet_pvc_optimizer = optim.RMSprop(self.UNet_pvc.parameters(), lr=self.learning_rate,momentum=0.9)
            self.double_optimizer = optim.RMSprop(list(self.UNet_denoiser.parameters())+list(self.UNet_pvc.parameters()), lr=self.learning_rate,momentum=0.9)
        else:
            raise ValueError("Unknown optimizer. Choose between : Adam, AdamW, SGD, RMSprop")

        self.learning_rate_policy_infos = self.params['lr_policy']
        if self.learning_rate_policy_infos[0] == 'multiplicative':
            mult_rate = self.learning_rate_policy_infos[1]
            self.update_lr_every = self.learning_rate_policy_infos[2]
            lbda = lambda epoch: mult_rate

            # self.scheduler_unet_denoiser = optim.lr_scheduler.MultiplicativeLR(self.unet_denoiser_optimizer, lbda)
            # self.scheduler_unet_pvc = optim.lr_scheduler.MultiplicativeLR(self.unet_pvc_optimizer, lbda)
            self.scheduler = optim.lr_scheduler.MultiplicativeLR(self.double_optimizer, lbda)
        elif self.learning_rate_policy_infos[0] =="reduceplateau":

            factor=self.learning_rate_policy_infos[1]
            patience=self.learning_rate_policy_infos[2]

            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.double_optimizer, 'min',
                                                                  factor=factor,patience=patience,
                                                                  min_lr=1e-6)
            self.update_lr_every=1

    def init_losses_(self):
        self.losses_params = {'recon_loss': self.params['recon_loss'],
                              'lambda_recon': self.params['lambda_recon'], 'device': self.device}
        if "edcc" in self.params['recon_loss']:
            recon_loss = []
            lamba_recons = []
            for loss_name,loss_lambda in zip(self.params['recon_loss'],self.params['lambda_recon']):
                if loss_name!="edcc":
                    recon_loss.append(loss_name)
                    lamba_recons.append(loss_lambda)
            self.losses_params_denoiser = {'recon_loss': recon_loss,
                              'lambda_recon': lamba_recons, 'device': self.device}
        else:
            self.losses_params_denoiser = self.losses_params

        if (("poisson" in self.params) and self.params["poisson"]):
            self.losses_params_denoiser = {'recon_loss': self.params['recon_loss']+['Poisson'],
                                           'lambda_recon': self.params['lambda_recon']+[1], 'device': self.device}

        if "sure_poisson" in self.params['recon_loss']:
            recon_loss = []
            lamba_recons = []
            for loss_name,loss_lambda in zip(self.params['recon_loss'],self.params['lambda_recon']):
                if loss_name!="sure_poisson":
                    recon_loss.append(loss_name)
                    lamba_recons.append(loss_lambda)
            self.losses_params_pvc = {'recon_loss': recon_loss,
                              'lambda_recon': lamba_recons, 'device': self.device}
        else:
            self.losses_params_pvc = self.losses_params

        self.losses_denoiser = losses.UNetLosses(self.losses_params_denoiser)
        self.losses_pvc = losses.UNetLosses(self.losses_params_pvc)

    def init_losses(self):
        self.losses_params_denoiser = {'recon_loss': self.params['loss_denoiser'],
                              'lambda_recon': self.params['lambda_denoiser'], 'device': self.device}

        self.losses_params_pvc = {'recon_loss': self.params['loss_pvc'],
                              'lambda_recon': self.params['lambda_pvc'], 'device': self.device}

        self.losses_denoiser = losses.UNetLosses(self.losses_params_denoiser)
        self.losses_pvc = losses.UNetLosses(self.losses_params_pvc)

    def input_data(self, batch_inputs, batch_targets):
        self.truePVE_noisy = batch_inputs['PVE_noisy']
        self.truePVE_noisy_raw = batch_inputs['PVE_noisy']
        self.truePVE = batch_targets['PVE']
        self.truePVfree = batch_targets['PVfree']

        if self.with_rec_fp:
            self.true_rec_fp = batch_inputs['rec_fp']
        if self.with_att:
            self.attmap_fp = batch_inputs['attmap_fp']
        if self.with_lesion:
            self.lesion_mask_fp = batch_targets['lesion_mask']
        else:
            self.lesion_mask_fp = None

        # print('--------------')
        # print(f"max PVE_noisy : {torch.amax(self.truePVE_noisy, dim=(1, 2, 3))}")
        # print(f"max PVE : {torch.amax(self.truePVE, dim=(1, 2, 3))}")
        # print(f"max PVfree : {torch.amax(self.truePVfree, dim=(1, 2, 3))}")
        # print(f"max rec_fp : {torch.amax(self.true_rec_fp, dim=(1, 2, 3))}")
        # print(f"max att : {torch.amax(self.attmap_fp, dim=(1, 2, 3))}")

        self.normalize_data()
        # print(f"max PVE_noisy normed: {torch.amax(self.truePVE_noisy, dim=(1, 2, 3))}")
        # print(f"max rec_fp normed: {torch.amax(self.true_rec_fp, dim=(1, 2, 3))}")
        # print(f"max att normed: {torch.amax(self.attmap_fp, dim=(1, 2, 3))}")

    def normalize_data(self):
        if self.params['data_normalisation'] == "3d_max":
            if self.with_rec_fp:
                self.norm = self.true_rec_fp.amax((1, 2, 3))
            else:
                self.norm = self.truePVE_noisy.amax((1, 2, 3))

            self.input_max = self.truePVE_noisy.amax((1, 2, 3))

            self.norm[self.norm == 0] = 1  # avoids nan after division by max

            self.truePVE_noisy = self.truePVE_noisy / self.norm[:, None, None, None]
            if self.with_att:
                max_attmap = torch.amax(self.attmap_fp, dim=(1, 2, 3))
                max_attmap[max_attmap == 0] = 1  # avoids nan after division by max
                self.attmap_fp = self.attmap_fp / max_attmap[:, None, None, None]

            if self.with_rec_fp:
                self.true_rec_fp = self.true_rec_fp / torch.amax(self.true_rec_fp, dim=(1, 2, 3))[:, None, None, None]

        elif self.params['data_normalisation'] in ["3d_sum", "3d_softmax"]:
            self.norm = self.truePVE_noisy.sum((1, 2, 3))
            self.input_max = self.truePVE_noisy.amax((1, 2, 3))[:, None, None, None]

            self.truePVE_noisy = self.truePVE_noisy / self.input_max
            if self.with_att:
                max_attmap = torch.amax(self.attmap_fp, dim=(1, 2, 3))
                max_attmap[max_attmap == 0] = 1  # avoids nan after division by max
                self.attmap_fp = self.attmap_fp / max_attmap[:, None, None, None]
            if self.with_rec_fp:
                self.true_rec_fp = self.true_rec_fp / self.input_max
        elif self.params['data_normalisation']=="sino_sum":
            self.norm = self.truePVE_noisy.sum((2, 3))
            self.input_max = self.truePVE_noisy.amax((1, 2, 3))[:, None, None, None]

            self.truePVE_noisy = self.truePVE_noisy / self.input_max
            if self.with_att:
                max_attmap = torch.amax(self.attmap_fp, dim=(1, 2, 3))
                max_attmap[max_attmap == 0] = 1  # avoids nan after division by max
                self.attmap_fp = self.attmap_fp / max_attmap[:, None, None, None]
            if self.with_rec_fp:
                # self.true_rec_fp = self.true_rec_fp / self.input_max
                max_rec_fp = torch.amax(self.true_rec_fp, dim=(1, 2, 3))
                max_rec_fp[max_rec_fp == 0] = 1  # avoids nan after division by max
                self.true_rec_fp = self.true_rec_fp / max_rec_fp[:,None,None,None]
        elif self.params['data_normalisation']=="act_cons":
            self.norm = self.truePVE_noisy.sum((2, 3))
        else:
            self.norm = None

    def denormalize_data_denoiser(self):
        if self.params['data_normalisation'] == "3d_max":
            self.fakePVE = self.fakePVE * self.input_max[:,None,None,None,None] / self.fakePVE.amax((1,2,3,4))[:,None,None,None,None]
        elif self.params['data_normalisation'] == "3d_sum":
            self.fakePVE = (self.fakePVE / self.fakePVE.sum((1, 2, 3, 4))[:,None,None,None,None]) * self.norm[:,None,None,None,None]
        elif self.params['data_normalisation'] in ["sino_sum","act_cons"]:
            projs_sum = self.fakePVE.sum((3, 4))
            projs_sum[projs_sum==0] = 1
            self.fakePVE = (self.fakePVE / projs_sum[:,:,:,None,None]) * self.norm[:, None, :, None, None]
        elif self.params['data_normalisation'] == "3d_softmax":
            self.fakePVE = torch.exp(self.fakePVE) / torch.exp(self.fakePVE).sum((1, 2, 3, 4))[:, None, None,
                                                           None, None] * self.norm[:, None, None, None, None]

    def normalize_data_pvc(self):
        if self.params['data_normalisation'] == "3d_max":
            self.fakePVE = self.fakePVE / self.norm[:,None, None, None, None]
        elif self.params['data_normalisation'] in ["3d_sum", "3d_softmax", "sino_sum"]:
            # self.fakePVE = self.fakePVE / self.input_max[:,:,:,:,None]
            max = self.fakePVE.amax((1,2,3,4))
            max_ = max.clone()
            max_[max==0] = 1
            self.fakePVE = self.fakePVE / max_[:,None,None,None,None]

    def denormalize_data_pvc(self):
        if self.params['data_normalisation'] == "3d_max":
            self.fakePVfree = self.fakePVfree * self.norm[:, None, None, None, None] / self.fakePVfree.amax((1,2,3,4))[:,None,None,None,None]
        elif self.params['data_normalisation'] == "3d_sum":
            self.fakePVfree = (self.fakePVfree / self.fakePVfree.sum((1, 2, 3, 4))[:, None, None, None,
                                                 None]) * self.norm[:, None, None, None, None]
        elif self.params['data_normalisation'] in ["sino_sum","act_cons"]:
            projs_sum = self.fakePVfree.sum((3, 4))
            projs_sum[projs_sum==0] = 1
            self.fakePVfree = (self.fakePVfree / projs_sum[:,:,:,None,None]) * self.norm[:,None,:,None, None]

        elif self.params['data_normalisation'] == "3d_softmax":
            self.fakePVfree = torch.exp(self.fakePVfree) / torch.exp(self.fakePVfree).sum((1, 2, 3, 4))[:, None, None,
                                                           None, None] * self.norm[:, None, None, None, None]

        # fig,ax = plt.subplots()
        # ax.plot((self.truePVE_noisy.detach().cpu().numpy())[0,:,:,:].sum((1,2)),label="PVEnoisy")
        # ax.plot((self.fakePVE.detach().cpu().numpy())[0,0,:,:,:].sum((1,2)),label="fakePVE")
        # ax.plot((self.truePVfree.detach().cpu().numpy())[0,:,:,:].sum((1,2)),label="truePVfree")
        # ax.plot((self.fakePVfree.detach().cpu().numpy())[0,0,:,:,:].sum((1,2)),label="fakePVfree")
        # ax.plot((self.true_rec_fp.detach().cpu().numpy())[0,:,:,:].sum((1,2)),label="recfp")
        # ax.legend()
        # plt.show()

    def forward_unet_denoiser(self):
        if self.dim==2:
            input_denoiser = torch.concat((self.truePVE_noisy,self.true_rec_fp),dim=1) if self.with_rec_fp else self.truePVE_noisy
        elif self.dim==3:
            if self.with_att:
                input_denoiser = torch.concat((self.truePVE_noisy[:,None,:,:,:], self.true_rec_fp[:,None,:,:,:], self.attmap_fp[:,None,:,:,:]), dim=1) if self.with_rec_fp else torch.concat((self.truePVE_noisy[:,None,:,:,:], self.attmap_fp[:,None,:,:,:]), dim=1)
            else:
                input_denoiser = torch.concat((self.truePVE_noisy[:,None,:,:,:], self.true_rec_fp[:,None,:,:,:]), dim=1) if self.with_rec_fp else self.truePVE_noisy[:,None,:,:,:]

        self.fakePVE = self.UNet_denoiser(input_denoiser)

        # print(f"max fakePVE normed: {torch.amax(self.fakePVE, dim=(1,2,3,4))}")
        self.denormalize_data_denoiser()
        # print(f"max fakePVE denormed: {torch.amax(self.fakePVE, dim=(1,2,3,4))}")

    def forward_unet_pvc(self):
        self.normalize_data_pvc()
        # print(f"max fakePVE re-normed: {torch.amax(self.fakePVE, dim=(1, 2, 3,4))}")

        if self.dim==2:
            input_pvc = torch.concat((self.fakePVE,self.true_rec_fp),dim=1) if self.with_rec_fp else self.fakePVE
        elif self.dim==3:
            if self.with_att:
                input_pvc = torch.concat((self.fakePVE,self.true_rec_fp[:,None,:,:,:], self.attmap_fp[:,None,:,:,:]), dim=1) if self.with_rec_fp else torch.concat((self.fakePVE, self.attmap_fp[:,None,:,:,:]), dim=1)
            else:
                input_pvc = torch.concat((self.fakePVE,self.true_rec_fp[:,None,:,:,:]), dim=1) if self.with_rec_fp else self.fakePVE

        # self.fakePVfree = self.UNet_pvc(input_pvc.detach())
        self.fakePVfree = self.UNet_pvc(input_pvc)
        # print(f"max fakePVfree normed: {torch.amax(self.fakePVfree, dim=(1, 2, 3,4))}")
        #
        self.denormalize_data_pvc()
        # print(f"max fakePVfree de-normed: {torch.amax(self.fakePVfree, dim=(1, 2, 3,4))}")


    def losses_unet_denoiser(self):
        self.unet_denoiser_loss = self.losses_denoiser.get_unet_loss(target=self.truePVE, output=self.fakePVE if self.dim==2 else self.fakePVE[:,0,:,:,:],lesion_mask=self.lesion_mask_fp,
                                                                     input_raw=self.truePVE_noisy_raw, model=self.UNet_denoiser)

    def losses_unet_pvc(self):
        self.unet_pvc_loss = self.losses_pvc.get_unet_loss(target=self.truePVfree,output=self.fakePVfree if self.dim==2 else self.fakePVfree[:,0,:,:,:], lesion_mask=self.lesion_mask_fp)

    def backward_unet_denoiser(self):
        if self.amp:
            self.scaler.scale(self.unet_denoiser_loss).backward(retain_graph=True)
            self.scaler.step(self.unet_denoiser_optimizer)
        else:
            self.unet_denoiser_loss.backward(retain_graph=True)
            self.unet_denoiser_optimizer.step()

    def backward_unet_pvc(self):
        if self.amp:
            self.scaler.scale(self.unet_pvc_loss).backward()
            self.scaler.step(self.unet_pvc_optimizer)
        else:
            self.unet_pvc_loss.backward()
            self.unet_pvc_optimizer.step()

    def backward_double_unet(self):
        # double_loss = self.unet_denoiser_loss + self.unet_pvc_loss
        if self.amp:
            if self.denoise:
                self.scaler.scale(self.unet_denoiser_loss + self.unet_pvc_loss).backward()
            else:
                self.scaler.scale(self.unet_pvc_loss).backward()

            # GRADIENT CLIPPING
            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.double_optimizer)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(list(self.UNet_denoiser.parameters())+list(self.UNet_pvc.parameters()), 1)

            self.scaler.step(self.double_optimizer)
            self.scaler.update()
            self.double_optimizer.zero_grad(set_to_none=True)
        else:
            if self.denoise:
                (self.unet_denoiser_loss + self.unet_pvc_loss).backward()
            else:
                self.unet_pvc_loss.backward()
            self.double_optimizer.step()
            self.double_optimizer.zero_grad(set_to_none=True)

    def forward(self, batch):
        self.truePVE_noisy = batch['PVE_noisy']
        if self.with_rec_fp:
            self.true_rec_fp = batch['rec_fp']
        if self.with_att:
            self.attmap_fp = batch['attmap_fp']

        self.normalize_data()
        with autocast(enabled=self.amp, dtype=torch.float16):
            self.forward_unet_denoiser()
            self.forward_unet_pvc()
        self.fakePVfree = self.fakePVfree[:, 0, :, :, :]
        return self.fakePVfree

    def optimize_parameters(self):
        # Unet denoiser update
        self.set_requires_grad(self.UNet_denoiser, requires_grad=True)
        self.set_requires_grad(self.UNet_pvc, requires_grad=True)
        # self.double_optimizer.zero_grad(set_to_none=True)
        # self.unet_denoiser_optimizer.zero_grad(set_to_none=True)
        # with autocast(enabled=self.amp):
            # self.forward_unet_denoiser()
            # self.losses_unet_denoiser()
        # self.backward_unet_denoiser()

        # Unet pvc update
        # self.set_requires_grad(self.UNet_pvc, requires_grad=True)
        # self.unet_pvc_optimizer.zero_grad(set_to_none=True)
        # with autocast(enabled=self.amp):
        #     self.forward_unet_pvc()
            # self.losses_unet_pvc()
        # self.backward_unet_pvc()

        with autocast(enabled=self.amp, dtype=torch.float16):
            self.forward_unet_denoiser()
            if self.denoise:
                self.losses_unet_denoiser()
            self.forward_unet_pvc()
            self.losses_unet_pvc()
        self.backward_double_unet()

        if self.denoise:
            self.mean_unet_denoiser_loss += self.unet_denoiser_loss.item()
        self.mean_unet_pvc_loss += self.unet_pvc_loss.item()
        self.current_iteration += 1

        # self.iter_loss.append((self.unet_denoiser_loss + self.unet_pvc_loss).item())

        self.del_variables()


    def del_variables(self):
        if self.with_rec_fp:
            if self.with_att:
                del self.truePVE_noisy,self.true_rec_fp,self.attmap_fp
            else:
                del self.truePVE_noisy,self.true_rec_fp
        else:
            if self.with_att:
                del self.truePVE_noisy,self.attmap_fp
            else:
                del self.truePVE_noisy

        if self.with_lesion:
            del self.truePVE, self.truePVfree,self.lesion_mask_fp
        else:
            del self.truePVE, self.truePVfree
            del self.lesion_mask_fp

        del self.fakePVE, self.fakePVfree

    def update_epoch(self): 
        self.unet_denoiser_losses.append(self.mean_unet_denoiser_loss / self.current_iteration)
        self.unet_pvc_losses.append(self.mean_unet_pvc_loss / self.current_iteration)
        if self.verbose > 1:
            print(f'Unet denoiser loss : {round(self.unet_denoiser_losses[-1], 5)} / pvc loss : {round(self.unet_pvc_losses[-1], 5)}')

        if self.current_epoch % self.update_lr_every == 0:
            # self.scheduler_unet_denoiser.step()
            # self.scheduler_unet_pvc.step()
            if self.learning_rate_policy_infos[0]=="multiplicative":
                self.scheduler.step()
            elif self.learning_rate_policy_infos[0]=="reduceplateau":
                self.scheduler.step(self.test_error[-1][1])

        if self.verbose > 1:
            # print(f'next lr : {self.scheduler_unet_denoiser.get_last_lr()}')
            # print(f'next lr : {self.scheduler.get_last_lr() }')
            print(f'next lr : {self.double_optimizer.param_groups[0]["lr"] }')

        self.current_epoch += 1
        self.current_iteration = 0
        self.mean_unet_denoiser_loss, self.mean_unet_pvc_loss = 0,0

        # fig,ax = plt.subplots()
        # ax.plot(self.iter_loss)
        # plt.show()
        # self.iter_loss=[]

    def save_model(self, output_path=None, save_json=False):
        self.params['start_epoch'] = self.start_epoch
        self.params['current_epoch'] = self.current_epoch
        jean_zay = self.params['jean_zay']

        if output_path is None:
            if self.output_folder:
                output_path = os.path.join(self.output_folder, self.output_pth)
            else:
                raise ValueError("Error: no output_folder specified")

        torch.save({'saving_date': time.asctime(),
                    'epoch': self.current_epoch,
                    'unet_denoiser': self.UNet_denoiser.module.state_dict() if jean_zay else self.UNet_denoiser.state_dict(),
                    'unet_pvc': self.UNet_pvc.module.state_dict() if jean_zay else self.UNet_pvc.state_dict(),
                    # 'unet_denoiser_opt': self.unet_denoiser_optimizer.state_dict(),
                    # 'unet_pvc_opt': self.unet_pvc_optimizer.state_dict(),
                    'double_optimizer': self.double_optimizer.state_dict(),
                    'unet_denoiser_losses': self.unet_denoiser_losses,
                    'unet_pvc_losses': self.unet_pvc_losses,
                    'test_error': self.test_error,
                    'val_mse':self.val_error_MSE,
                    'val_mae': self.val_error_MAE,
                    'params': self.params
                    }, output_path)
        if self.verbose > 0:
            print(f'Model saved at : {output_path}')

        if save_json:
            output_json = output_path[:-4] + '.json'
            formatted_params = self.format_params()
            jsonFile = open(output_json, "w")
            jsonFile.write(formatted_params)
            jsonFile.close()

    def load_model(self, pth_path, new_lr=None):
        if self.verbose > 0:
            print(f'Loading Model from {pth_path}... ')
        checkpoint = torch.load(pth_path, map_location=self.device)

        if hasattr(self.UNet_denoiser, 'module'):
            self.UNet_denoiser.module.load_state_dict(checkpoint['unet_denoiser'])
            self.UNet_pvc.module.load_state_dict(checkpoint['unet_pvc'])
        else:
            self.UNet_denoiser.load_state_dict(checkpoint['unet_denoiser'])
            self.UNet_pvc.load_state_dict(checkpoint['unet_pvc'])
        self.unet_denoiser_losses,self.unet_pvc_losses = checkpoint['unet_denoiser_losses'],checkpoint['unet_pvc_losses']
        self.test_error = checkpoint['test_error']

        self.val_error_MSE = checkpoint['val_mse'] if 'val_mse' in checkpoint else []
        self.val_error_MAE = checkpoint['val_mae'] if 'val_mae' in checkpoint else []
        self.current_epoch = checkpoint['epoch']

        if self.resume_training:
            if (new_lr is not None):
                print(f"NEW LEARNING RATE FOR RESUME TRAINING : {self.learning_rate}")
            else:
                self.learning_rate = checkpoint['double_optimizer']['param_groups'][0]['lr']

            self.init_optimization()
            self.init_losses()
            self.double_optimizer.load_state_dict(checkpoint['double_optimizer'])
            for g in self.double_optimizer.param_groups:
                g['lr'] = self.learning_rate
            self.start_epoch = self.current_epoch

    def switch_eval(self):
        self.UNet_denoiser.eval()
        self.UNet_pvc.eval()

    def switch_train(self):
        self.UNet_denoiser.train()
        self.UNet_pvc.train()

    def switch_device(self, device):
        self.device = device
        self.UNet_denoiser = self.UNet_denoiser.to(device=device)
        self.UNet_pvc = self.UNet_pvc.to(device=device)

    def show_infos(self, mse=False):
        formatted_params = self.format_params()
        print('*' * 80)
        print('PARAMETRES (json param file) : \n')
        print(formatted_params)
        print('*' * 80)

        print('MEAN SQUARE ERROR on TEST DATA')
        print(f'list of errors on test data :{self.test_error} ')
        print('*' * 20 + "DENOISER" + '*'*20)
        print(self.UNet_denoiser)
        nb_params = sum(p.numel() for p in self.UNet_denoiser.parameters())
        print(f'NUMBER OF PARAMERS : {nb_params}')
        self.nb_params = nb_params
        print('*' * 20 + "PVC" + '*'*20)
        print(self.UNet_pvc)
        nb_params = sum(p.numel() for p in self.UNet_pvc.parameters())
        print(f'NUMBER OF PARAMERS : {nb_params}')
        if hasattr(self, "losses_denoiser"):
            print('Losses : ')
            print("denoiser: ", self.losses_params_denoiser)
            print("pvc: ", self.losses_params_pvc)
            print('Denoiser loss : ')
            print(self.losses_denoiser)
            print('PVC loss : ')
            print(self.losses_pvc)
            print('*' * 80)
        self.nb_params += nb_params

        if self.params['jean_zay']==False:
            from torchscan import summary
            summary(module = self.UNet_denoiser,input_shape=(3,128,80,112),receptive_field=True)
            summary(module = self.UNet_pvc,input_shape=(3,128,80,112),receptive_field=True)


    def plot_losses(self, save, wait, title):
        plots.plot_losses_UNet(unet_losses=self.unet_denoiser_losses, test_mse=[], save=save, wait=True, title=title)
        plots.plot_losses_UNet(unet_losses=self.unet_pvc_losses, test_mse=[], save=save, wait=wait, title=title)

        fig,ax =plt.subplots()
        ax.plot(self.unet_denoiser_losses, label="denoiser")
        ax.plot(self.unet_pvc_losses, label="pvc")
        fig.legend()