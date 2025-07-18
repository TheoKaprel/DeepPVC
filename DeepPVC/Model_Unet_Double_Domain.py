import os.path
import time

import itk
import torch
from torch import optim

from . import networks, losses, helpers_data_parallelism, plots
from torch.cuda.amp import autocast, GradScaler

from .Model_base import ModelBase

import sys
import os
host = os.uname()[1]
if (host !='suillus'):
    sys.path.append("/linkhome/rech/gencre01/uyo34ub/WORK/PVE/eDCCs_torch")
else:
    sys.path.append("/export/home/tkaprelian/Desktop/eDCCsTorch")


from SPECTmodel import SPECT_system_torch
from projectors import ForwardProjection,BackProjection


class UNet_Double_Domain(ModelBase):
    def __init__(self, params, from_pth=None, resume_training=False, device=None):
        assert (params['network'] == 'double_domain')
        super().__init__(params, resume_training, device=device)
        self.network_type = 'double_domain'
        self.verbose = params['verbose']

        if "dim" in params:
            if params["dim"] == "2d":
                self.dim = 2
            elif params["dim"] == "3d":
                self.dim = 3
        else:
            self.dim = 2

        self.init_feature_kernel = params['init_feature_kernel']
        self.nb_ed_layers = params['nb_ed_layers']
        if "ed_blocks" in params:
            self.ed_blocks = params["ed_blocks"]
        else:
            self.ed_blocks = "conv-relu-norm"

        self.final_feature_kernel = params['final_feature_kernel'] if "final_feature_kernel" in params else 3
        self.kernel_size = params['kernel_size'] if "kernel_size" in params else 3

        self.with_rec_fp = params['with_rec_fp']
        self.with_PVCNet_rec = params["with_PVCNet_rec"] if "with_PVCNet_rec" in params else False

        self.hidden_channels_unet = params["hidden_channels_unet"]
        self.unet_activation = params["unet_activation"]

        self.layer_norm = params['layer_norm']
        self.residual_layer = params['residual_layer']
        self.ResUnet = params['resunet']

        if self.residual_layer:
            self.residual_channel = 0
        else:
            self.residual_channel = -1

        self.final_2dconv = True if (params['dim'] == "3d" and params['inputs'] == "projs") else False

        if "archi" not in params:
            self.archi = "unet"
        else:
            self.archi = params["archi"]

        self.paths = params['paths'] if "paths" in params else False

        self.img_to_img = (len(params["sino_loss"]) == 0) or (len(params["sino_loss"])==1 and (params["sino_loss"][0]=="consistency"))
        self.null_space = (len(params["sino_loss"]) == 0 and len(params["img_loss"]) == 1 and params["img_loss"][0] == "null_space")


        if from_pth:
            self.for_training = False
            self.init_model()
            self.init_losses()
            self.init_spect_recons()
            if self.verbose > 1:
                print(
                    'normalement self.load_model(from_pth) mais là non, on le fait juste apres l initialisation des gpus etc')
        else:
            self.for_training = True
            self.init_model()
            self.init_optimization()
            self.init_losses()

            self.init_spect_recons()

            self.unet_losses = []
            self.test_error = []

            self.current_epoch = 1
            self.start_epoch = 1

        self.current_iteration = 0
        self.mean_unet_loss = 0

        self.amp = self.params['amp']
        if self.amp:
            self.scaler = GradScaler()
        self.autocat_losses = self.amp

    def init_model(self):
        if self.verbose > 0:
            print(f'models device is supposed to be : {self.device}')

        if self.archi == "unet_sym":
            self.UNet_sino = networks.UNet_symetric(input_channel=self.input_channels, ngc=self.hidden_channels_unet,
                                               dim=self.dim, init_feature_kernel=self.init_feature_kernel,
                                               final_feature_kernel=self.final_feature_kernel,
                                               nb_ed_layers=self.nb_ed_layers,
                                               output_channel=self.output_channels,
                                               generator_activation=self.unet_activation,
                                               use_dropout=self.use_dropout, leaky_relu=self.leaky_relu,
                                               norm=self.layer_norm, residual_layer=self.residual_channel,
                                               blocks=self.ed_blocks,
                                               ResUnet=self.ResUnet,
                                               final_2dconv=self.final_2dconv, final_2dchannels=2 * self.params['nb_adj_angles'] if self.final_2dconv else 0,
                                               paths=self.paths,
                                               kernel_size=self.kernel_size
                                               ).to(device=self.device) if not (self.img_to_img or self.null_space) else torch.nn.Identity()

            self.UNet_img = networks.UNet_symetric(input_channel=self.input_channels, ngc=self.hidden_channels_unet,
                                               dim=self.dim, init_feature_kernel=self.init_feature_kernel,
                                               final_feature_kernel=self.final_feature_kernel,
                                               nb_ed_layers=self.nb_ed_layers,
                                               output_channel=self.output_channels,
                                               generator_activation=self.unet_activation,
                                               use_dropout=self.use_dropout, leaky_relu=self.leaky_relu,
                                               norm=self.layer_norm, residual_layer=self.residual_channel,
                                               blocks=self.ed_blocks,
                                               ResUnet=self.ResUnet,
                                               final_2dconv=self.final_2dconv, final_2dchannels=2 * self.params['nb_adj_angles'] if self.final_2dconv else 0,
                                               paths=self.paths,
                                               kernel_size=self.kernel_size
                                               ).to(device=self.device)

        if self.for_training:
            if "init" not in self.params:
                self.params["init"] = "none"
            self.UNet_img = networks.init_weights(net=self.UNet_img, init_type=self.params["init"])
            self.UNet_sino = networks.init_weights(net=self.UNet_sino, init_type=self.params["init"])

        if self.params['jean_zay']:
            helpers_data_parallelism.init_data_parallelism(model=self)

        if (('compile' in self.params) and (self.params['compile'] == True)):
            self.compile = True
        else:
            self.compile = False

        if (self.compile and self.for_training):
            self.UNet_img = torch.compile(self.UNet_img)
            self.UNet_sino = torch.compile(self.UNet_sino)

    def init_spect_recons(self):
        spect_data_folder = self.params["spect_data_folder"]
        if not self.img_to_img:
            self.nsubsets = self.params["nsubsets"]
            self.spect_model = SPECT_system_torch(projections_fn=os.path.join(spect_data_folder, "projs_rtk_PVE_noisy.mha"),
                                       like_fn=os.path.join(spect_data_folder, "IEC_BG_attmap_cropped_rot_4mm.mhd"),
                                       fbprojectors="JosephAttenuated",
                                       attmap_fn=os.path.join(spect_data_folder,"IEC_BG_attmap_cropped_rot_4mm.mhd"),
                                       nsubsets=self.nsubsets)
            if "nosem" in self.params:
                self.nosem = self.params["nosem"]
            else:
                self.nosem = 2
            self.bp_ones = []
            for subset in range(self.nsubsets):
                self.spect_model.set_geometry(subset)
                self.bp_ones.append(torch.tensor(self.spect_model.get_bp_ones(), device = self.device))

        # print("###############################################################################")
        # print("###############################################################################")
        # self.rtk_back_projection = lambda input_projs: BackProjection.apply(input_projs, self.spect_model)
        # check = torch.autograd.gradcheck(self.rtk_back_projection,
        #                                            100*torch.rand(self.spect_model.projection_array[self.spect_model.subset_ids,:,:].shape,dtype=torch.double,requires_grad=True),
        #                                            fast_mode = True)
        #
        # print(f"GRAD CHEKC FOR BACKPROJ: {check}")
        # print("###############################################################################")
        # print("###############################################################################")


        if ("consistency" in self.losses_sino_names or self.null_space):
            self.spect_model_for_consistency = SPECT_system_torch(projections_fn=os.path.join(spect_data_folder, "projs_rtk_PVE_noisy.mha"),
                                   like_fn=os.path.join(spect_data_folder, "IEC_BG_attmap_cropped_rot_4mm.mhd"),
                                   fbprojectors="Zeng" if self.img_to_img else "JosephAttenuated",
                                   attmap_fn=os.path.join(spect_data_folder,"IEC_BG_attmap_cropped_rot_4mm.mhd"),
                                   nsubsets=1)
            self.rtk_forward_projection_for_consistency = lambda input_img: ForwardProjection.apply(input_img, self.spect_model_for_consistency)

    def init_optimization(self):
        if self.optimizer == 'Adam':
            self.double_optimizer = optim.Adam(list(self.UNet_sino.parameters()) + list(self.UNet_img.parameters()), lr=self.learning_rate,
                                               weight_decay=self.weight_decay)
        else:
            raise ValueError("Unknown optimizer. Choose between : Adam, AdamW, SGD, RMSprop")

        self.learning_rate_policy_infos = self.params['lr_policy']
        if self.learning_rate_policy_infos[0] == 'multiplicative':
            mult_rate = self.learning_rate_policy_infos[1]
            self.update_lr_every = self.learning_rate_policy_infos[2]
            lbda = lambda epoch: mult_rate
            self.scheduler = optim.lr_scheduler.MultiplicativeLR(self.double_optimizer, lbda)
        elif self.learning_rate_policy_infos[0] == "reduceplateau":
            factor = self.learning_rate_policy_infos[1]
            patience = self.learning_rate_policy_infos[2]

            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.double_optimizer, 'min',
                                                                  factor=factor, patience=patience,
                                                                  min_lr=1e-6)
            self.update_lr_every = 1

    def init_losses(self):
        self.losses_img = [losses.get_nn_loss(loss_name=loss_name) for loss_name in self.params["img_loss"]]
        self.losses_sino = [losses.get_nn_loss(loss_name=loss_name) for loss_name in self.params["sino_loss"]]

        self.losses_sino_names = self.params["sino_loss"]
        self.losses_img_names = self.params["img_loss"]

        self.losses_img_lambdas = self.params["img_loss_lambda"]
        self.losses_sino_lambdas = self.params["sino_loss_lambda"]

    def input_data(self, batch_inputs, batch_targets):
        if not self.null_space:
            self.truePVE_noisy = batch_inputs['PVE_noisy']

        if (self.img_to_img or self.null_space):
            self.rec = batch_inputs["rec"][:,None,:,:,:]
        else:
            self.truePVfree = batch_targets['PVfree']

        self.true_src = batch_targets['src_4mm']

    def normalize_sino(self):
        self.norm_sino = self.truePVE_noisy.sum((2, 3))
        self.input_sino_max = self.truePVE_noisy.amax((1, 2, 3))[:, None, None, None]
        self.truePVE_noisy = self.truePVE_noisy / self.input_sino_max

    def denormalize_sino(self):
        projs_sum = self.fakePVfree.sum((3, 4))[:, :, :, None, None]
        projs_sum[projs_sum == 0] = 1
        self.fakePVfree = self.fakePVfree / projs_sum * self.norm_sino[:, None, :, None, None]

    def forward_sino(self):
        self.normalize_sino()
        input_sino = self.truePVE_noisy[:, None, :, :, :]
        self.fakePVfree = self.UNet_sino(input_sino)
        self.denormalize_sino()

    def recons(self):
        rec_k = torch.ones_like(self.bp_ones[0], device = self.device, requires_grad=True)

        for k in range(self.nosem):
            for subset in range(self.nsubsets):
                self.spect_model.set_geometry(subset)

                self.rtk_forward_projection = lambda input_img : ForwardProjection.apply(input_img, self.spect_model)
                self.rtk_back_projection = lambda input_projs : BackProjection.apply(input_projs, self.spect_model)

                rec_k_fp = self.rtk_forward_projection(rec_k)
                rec_k = rec_k / (self.bp_ones[subset]+1e-8) * self.rtk_back_projection(
                    self.fakePVfree[0,0,self.spect_model.subset_ids,:,:] / (rec_k_fp+1e-8))

        self.rec = rec_k[None,None,:,:,:]

    def normalize_img(self):
        self.norm_rec = self.rec.sum((1, 2, 3,4))
        self.max_rec = self.rec.amax((2, 3, 4))[:,:,None, None, None]
        self.rec_n = self.rec / self.rec.amax((2, 3, 4))[:,:,None, None, None]


    def denormalize_img(self):
        self.fake_src = (self.fake_src / self.fake_src.sum((1, 2, 3, 4))[:, None, None, None, None]) * self.norm_rec[:,None,None,None,None]

    def forward_img(self):
        self.normalize_img()
        self.fake_src = self.UNet_img(self.rec_n)
        self.fake_src = self.fake_src * self.max_rec
        # self.denormalize_img()
        if self.null_space:
            self.project_to_null_space()

    def project_to_null_space(self):
        fp_fake_src = self.rtk_forward_projection_for_consistency(self.fake_src[0,0,:,:,:])

        itk.imwrite(itk.image_from_array(self.fake_src[0,0,:,:,:].detach().cpu().numpy()), f"/export/home/tkaprelian/temp/null_space/x_theta_{self.current_epoch}.mha")

        rec_k = torch.ones_like(self.bp_ones[0], device = self.device, requires_grad=True)
        for k in range(self.nosem):
            for subset in range(self.nsubsets):
                self.spect_model.set_geometry(subset)

                self.rtk_forward_projection = lambda input_img : ForwardProjection.apply(input_img, self.spect_model)
                self.rtk_back_projection = lambda input_projs : BackProjection.apply(input_projs, self.spect_model)

                rec_k_fp = self.rtk_forward_projection(rec_k)
                rec_k = rec_k / (self.bp_ones[subset]+1e-8) * self.rtk_back_projection(
                    fp_fake_src[self.spect_model.subset_ids,:,:] / (rec_k_fp+1e-8))

        self.fake_src = torch.nn.functional.relu(self.rec[0,0,:,:,:] + self.fake_src[0,0,:,:,:] - rec_k)
        # self.fake_src = self.fake_src[0,0,:,:,:]

        itk.imwrite(itk.image_from_array(self.rec[0,0,:,:,:].detach().cpu().numpy()), f"/export/home/tkaprelian/temp/null_space/x_rec_{self.current_epoch}.mha")
        itk.imwrite(itk.image_from_array(rec_k[:,:,:].detach().cpu().numpy()), f"/export/home/tkaprelian/temp/null_space/x_rec_x_theta_{self.current_epoch}.mha")
        # if self.current_epoch%10==0:
        itk.imwrite(itk.image_from_array(self.fake_src.detach().cpu().numpy()), f"/export/home/tkaprelian/temp/null_space/fake_src_{self.current_epoch}.mha")

        self.fake_src = self.fake_src[None,None,:,:,:]

    def compute_losses(self):
        if "consistency" in self.losses_sino_names:
            fp_fake_src = self.rtk_forward_projection_for_consistency(self.fake_src[0,0,:,:,:])[None,:,:,:]
        else:
            fp_fake_src = None

        # img domain losses
        self.unet_loss = sum([lbda * loss(self.fake_src[:,0,:,:,:],self.true_src)
                              for lbda,loss in zip(self.losses_img_lambdas,self.losses_img)])

        # sino domain losses
        self.unet_loss += sum([lbda * loss(self.truePVfree, self.fakePVfree[:,0,:,:,:]) if name!="consistency" else
                           lbda * loss(fp_fake_src, self.truePVE_noisy)
                          for name,lbda,loss in zip(self.losses_sino_names,self.losses_sino_lambdas, self.losses_sino)])



    def backward(self):
        if self.amp:
            self.scaler.scale(self.unet_loss).backward()
            self.scaler.step(self.double_optimizer)
            self.scaler.update()
            self.double_optimizer.zero_grad(set_to_none=True)
        else:
            (self.unet_loss).backward()
            self.double_optimizer.step()
            self.double_optimizer.zero_grad(set_to_none=True)

    def forward(self, batch):
        if self.img_to_img:
            self.rec = batch["rec"][:, None, :, :, :]
        else:
            self.truePVE_noisy = batch['PVE_noisy']

        with autocast(enabled=self.amp, dtype=torch.float16):
            if not self.img_to_img:
                self.forward_sino()
                self.recons()
            self.forward_img()
        self.fake_src = self.fake_src[:, 0, :, :, :]
        return self.fake_src

    def optimize_parameters(self):
        self.double_optimizer.zero_grad(set_to_none=True)

        # Unet denoiser update
        self.set_requires_grad(self.UNet_sino, requires_grad=True)
        self.set_requires_grad(self.UNet_img, requires_grad=True)

        with autocast(enabled=self.amp, dtype=torch.float16):
            if (not self.img_to_img and not self.null_space):
                self.forward_sino()
                self.recons()
            self.forward_img()
            self.compute_losses()
        self.backward()

        self.mean_unet_loss += self.unet_loss.item()
        self.current_iteration += 1
        self.del_variables()

    def del_variables(self):
        if not self.null_space:
            del self.truePVE_noisy
        del self.rec
        del self.fake_src

        if (not self.img_to_img and not self.null_space):
            del self.fakePVfree
            del self.truePVfree

    def update_epoch(self):
        self.unet_losses.append(self.mean_unet_loss / self.current_iteration)
        if self.verbose > 1:
            print(f'Unet loss : {round(self.unet_losses[-1], 5)}')

        if self.current_epoch % self.update_lr_every == 0:
            if self.learning_rate_policy_infos[0] == "multiplicative":
                self.scheduler.step()
            elif self.learning_rate_policy_infos[0] == "reduceplateau":
                self.scheduler.step(self.test_error[-1][1])

        if self.verbose > 1:
            # print(f'next lr : {self.scheduler.get_last_lr()}')
            print(f'next lr : {self.double_optimizer.param_groups[0]["lr"]}')

        self.current_epoch += 1
        self.current_iteration = 0
        self.mean_unet_loss = 0

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
                    'unet_sino': self.UNet_sino.module.state_dict() if hasattr(self.UNet_sino, 'module') else self.UNet_sino.state_dict(),
                    'unet_img': self.UNet_img.module.state_dict() if jean_zay else self.UNet_img.state_dict(),
                    'double_optimizer': self.double_optimizer.state_dict(),
                    'unet_losses': self.unet_losses,
                    'test_error': self.test_error,
                    'val_mse': self.val_error_MSE,
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

        if hasattr(self.UNet_sino, 'module'):
            self.UNet_sino.module.load_state_dict(checkpoint['unet_sino'])
            self.UNet_img.module.load_state_dict(checkpoint['unet_img'])
        else:
            self.UNet_sino.load_state_dict(checkpoint['unet_sino'])
            self.UNet_img.load_state_dict(checkpoint['unet_img'])
        self.unet_losses = checkpoint['unet_losses']
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
        self.UNet_sino.eval()
        self.UNet_img.eval()

    def switch_train(self):
        self.UNet_sino.train()
        self.UNet_img.train()

    def switch_device(self, device):
        self.device = device
        self.UNet_sino = self.UNet_sino.to(device=device)
        self.UNet_img = self.UNet_img.to(device=device)

    def show_infos(self, mse=False):
        formatted_params = self.format_params()
        print('*' * 80)
        print('PARAMETRES (json param file) : \n')
        print(formatted_params)
        print('*' * 80)

        print('MEAN SQUARE ERROR on TEST DATA')
        print(f'list of errors on test data :{self.test_error} ')
        print('*' * 20 + "UNET sino" + '*' * 20)
        print(self.UNet_sino)
        nb_params = sum(p.numel() for p in self.UNet_sino.parameters())
        self.nb_params = nb_params
        print(f'NUMBER OF PARAMERS : {nb_params}')

        print('*' * 20 + "UNET img" + '*' * 20)
        print(self.UNet_img)
        nb_params = sum(p.numel() for p in self.UNet_img.parameters())
        self.nb_params = nb_params
        print(f'NUMBER OF PARAMERS : {nb_params}')

        print("*************************************************")
        if hasattr(self, "losses"):
            print('Losses : ')
            # print(self.losses_params)
            print('loss : ')
            print(self.losses)
            print('*' * 80)
        elif hasattr(self, "losses_img"):
            print("Losses: ")
            print(f"image-domain loss: {self.losses_img_lambdas} {self.losses_img}")
            print(f"sinogram-domain loss: {self.losses_sino_lambdas} {self.losses_sino}")


        print("*************************************************")
        # if self.params['jean_zay']==False:
        #     from torchscan import summary

        # summary(module = self.UNet,input_shape=(3,128,80,112),receptive_field=True)

    def plot_losses(self, save, wait, title):
        plots.plot_losses_UNet(unet_losses=self.unet_losses, test_mse=[], save=save, wait=wait, title=title)
