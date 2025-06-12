import os.path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.cuda.amp import custom_fwd
from . import helpers

import sys
import os
host = os.uname()[1]
if (host !='suillus'):
    sys.path.append("/linkhome/rech/gencre01/uyo34ub/WORK/PVE/eDCCs_torch")
else:
    sys.path.append("/export/home/tkaprelian/Desktop/eDCCsTorch")


from exponential_projections_torch import ExponentialProjectionsTorch




def get_nn_loss(loss_name):
    if loss_name=="L1":
        return nn.L1Loss()
    elif loss_name=="L2":
        return nn.MSELoss()
    elif loss_name in ["Poisson", "poisson", "consistency"]:
        return nn.PoissonNLLLoss(log_input=False,full=False)
    elif loss_name=="null_space":
        return nn.MSELoss()
    elif loss_name=='BCE':
        return nn.BCEWithLogitsLoss()
    elif loss_name=="Wasserstein":
        return Wasserstein_loss()
    elif loss_name=="Sum":
        return Sum_loss()
    elif loss_name=="SmoothL1":
        return nn.SmoothL1Loss()
    elif loss_name=="lesion":
        return nn.L1Loss()
    elif loss_name=="conv":
        return nn.L1Loss()
    elif loss_name=="edcc":
        print("eDCCs loss not implemented yet")
        exit(0)
    elif loss_name=="sure_poisson":
        return SurePoissonLoss()
    else:
        print(f'ERROR in loss name {loss_name}')
        exit(0)

class Lesion_loss(nn.Module):
    def __init__(self):
        super(Lesion_loss, self).__init__()
        self.l1=nn.L1Loss()

    @custom_fwd
    def forward(self, y_true, y_pred, lesion_mask):
        return self.l1(y_true[lesion_mask], y_pred[lesion_mask])

class SurePoissonLoss(nn.Module):
    def __init__(self, tau=1):
        super(SurePoissonLoss, self).__init__()
        self.tau = tau

    def forward(self, p_noisy, p_output, model, true_rec_fp,attmap_fp):
        # generate a random vector b
        b = torch.empty_like(p_noisy).uniform_()
        b = b > 0.5
        b = (2 * b - 1) * 1.0  # binary [-1, 1]

        y1 = p_output

        input_denoiser = torch.concat(((p_noisy + self.tau * b)[:, None, :, :, :], true_rec_fp[:, None, :, :, :],attmap_fp[:, None, :, :, :]), dim=1)

        y2 = model()

        loss_sure = (
            (y1 - p_noisy).pow(2)
            -  p_noisy
            + (2.0 / self.tau) * (b * p_noisy * (y2 - y1))
        )

        loss_sure = loss_sure.reshape(p_noisy.size(0), -1).mean(1)
        return loss_sure


class PoissonLikelihood_loss(nn.Module):
    def __init__(self):
        super(PoissonLikelihood_loss, self).__init__()

    @custom_fwd
    def forward(self, y_true,y_pred):
        eps = 1e-6
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_true = y_true.view(y_true.shape[0], -1)

        p_l = -y_true+y_pred*torch.log(y_true + eps)-torch.lgamma(y_pred+1)
        return -torch.mean(p_l)


class gradient_penalty(nn.Module):
    # cf https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
    def __init__(self,device):
        super(gradient_penalty, self).__init__()
        self.device = device

    def switch_device(self,device):
        self.device=device

    @custom_fwd
    def forward(self,interpolates, model_interpolates):
        grad_outputs = torch.ones(model_interpolates.size(),device=self.device, requires_grad=False)

        gradients = torch.autograd.grad(
            outputs=model_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        return gradient_penalty


class Wasserstein_loss(nn.Module):
    def __init__(self):
        super(Wasserstein_loss, self).__init__()

    @custom_fwd
    def forward(self, D_disc,target):
        return (torch.mean( (-2*target+1) * D_disc))

class Sum_loss(nn.Module):
    def __init__(self):
        super(Sum_loss, self).__init__()

    @custom_fwd
    def forward(self,target,output):
        return torch.abs(target.sum() - output.sum())


class Model_Loss(nn.Module):
    def __init__(self, losses_params):
        super().__init__()
        self.loss_name, self.recon_loss,self.lambdas=[],[],[]
        if type(losses_params['recon_loss'])==list:
            for (loss,lbda) in zip(losses_params['recon_loss'],losses_params['lambda_recon']):
                if loss=="edcc":
                    edccs_data_folder = losses_params["edccs_data_folder"]
                    exp_projections = ExponentialProjectionsTorch(projs_fn=os.path.join(edccs_data_folder, "projs_rtk.mha"),
                                                                  attmap_fn=os.path.join(edccs_data_folder, "attmap.mha"),
                                                                  kregion_fn=os.path.join(edccs_data_folder, "background.mha"),
                                                                  conversion_factor_fn=os.path.join(edccs_data_folder, "conversion_factor.mha"),
                                                                  geometry_fn=os.path.join(edccs_data_folder, "geom_280.xml"),
                                                                  device_name="gpu")
                    em_slice = [32, 95]
                    edcc_loss = lambda p : exp_projections.compute_edcc_vectorized_input_proj(em_slice=em_slice,projections_tensor=p,del_mask=True,compute_var=True).mean()
                    self.recon_loss.append(edcc_loss)
                else:
                    self.recon_loss.append(get_nn_loss(loss_name=loss))
                self.lambdas.append(lbda)
                self.loss_name.append(loss)
        else:
            self.recon_loss=[get_nn_loss(loss_name=losses_params['recon_loss'])]
            self.lambdas=[1]
            self.loss_name = [losses_params['recon_loss']]

        self.device = helpers.get_auto_device(losses_params['device'])

    def extra_repr(self):
        extra_repr_str=''
        extra_repr_str+='(recon_loss):'
        for loss in self.recon_loss:
            mod_str = repr(loss)
            extra_repr_str+='  '+mod_str+','
        extra_repr_str+='\n'
        extra_repr_str+='(lambdas_recon):'
        for lbda in self.lambdas:
            extra_repr_str+='  ' + str(lbda)+','

        return extra_repr_str


class Pix2PixLosses(Model_Loss):
    def __init__(self, losses_params):
        super().__init__(losses_params=losses_params)
        self.adv_loss = get_nn_loss(loss_name=losses_params['adv_loss'])

        if losses_params['gradient_penalty']:
            self.gradient_penalty = gradient_penalty(device=self.device)

        self.ones = torch.tensor([0.0],device=self.device)

    def get_adv_loss(self):
        return self.adv_loss
    def get_recon_loss(self,target,output):
        weighted_recon_loss=sum([lbda * loss(target,output) for (lbda,loss) in zip(self.lambdas,self.recon_loss)])
        return weighted_recon_loss

    def get_gen_loss(self, disc_fake_hat, truePVfree, fakePVfree):
        # gen_adv_loss = self.adv_loss(disc_fake_hat, torch.ones_like(disc_fake_hat))
        gen_adv_loss = self.adv_loss(disc_fake_hat, self.ones.expand_as(disc_fake_hat))
        gen_rec_loss = self.get_recon_loss(truePVfree, fakePVfree)
        gen_loss = gen_adv_loss + gen_rec_loss
        return gen_loss

    def get_gradient_penalty(self,Discriminator,real,fake,condition):
        alpha = torch.randn((real.size(0), 1, 1, 1), device=self.device)
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
        model_interpolates = Discriminator(interpolates,condition)
        return self.gradient_penalty(interpolates, model_interpolates)


class UNetLosses(Model_Loss):
    def __init__(self, losses_params):
        super().__init__(losses_params)

    def get_unet_loss(self, target, output, lesion_mask=None, conv_psf=None, input_rec=None, input_raw = None,model=None):
        unet_loss = torch.tensor([0.], device=self.device)
        loss_val = 0
        for (loss_name,loss,lbda) in zip(self.loss_name,self.recon_loss,self.lambdas):
            if loss_name=="lesion":
                unet_loss+= lbda * loss(target[lesion_mask], output[lesion_mask])
            elif loss_name=="conv":
                unet_loss+= lbda * loss(input_rec[:,None,:,:,:], conv_psf(output[:,None,:,:,:]))
            elif loss_name=="Poisson":
                print(output[0,40,63,63], input_raw[0,40,63,63], target[0,40,63,63])
                # unet_loss+= lbda * loss(output, input_raw)
                unet_loss+= lbda * loss(output, target)
            elif loss_name=="poisson":
                unet_loss+=lbda * loss(output,target)
            elif loss_name=="sure_poisson":
                unet_loss+=lbda*loss(p_noisy=input_raw, p_output=output, model=model)
            elif loss_name=="edcc":
                # unet_loss+=lbda*loss(output)
                loss_val = torch.mean(torch.tensor([loss(output[k,:,:,:]) for k in range(output.shape[0])],device=output.device))
                unet_loss+= lbda*loss_val
            else:
                # unet_loss+= lbda * loss(target, output)
                loss_val = loss(output, target)
                unet_loss+= lbda * loss_val

            # print(f"{loss_name}: {loss_val}")
        return unet_loss
