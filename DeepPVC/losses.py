import torch
from torch import nn


def get_nn_loss(loss_name):
    if loss_name=="L1":
        return nn.L1Loss()
    elif loss_name=="L2":
        return nn.MSELoss()
    elif loss_name=="Poisson":
        return PoissonLikelihood_loss()
    elif loss_name=='BCE':
        return nn.BCEWithLogitsLoss()
    elif loss_name=="Wasserstein":
        return Wasserstein_loss()
    else:
        print(f'ERROR in loss name {loss_name}')
        exit(0)


class PoissonLikelihood_loss(nn.Module):
    def __init__(self):
        super(PoissonLikelihood_loss, self).__init__()

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

    def forward(self, D_disc,target):
        return (torch.mean( (-2*target+1) * D_disc))



class Pix2PixLosses:
    def __init__(self, losses_params):
        self.adv_loss = get_nn_loss(loss_name=losses_params['adv_loss'])

        self.recon_loss = get_nn_loss(loss_name=losses_params['recon_loss'])

        self.lambda_recon = losses_params['lambda_recon']
        self.device = losses_params['device']

    def get_adv_loss(self):
        return self.adv_loss
    def get_recon_loss(self):
        return self.recon_loss

    def get_gen_loss(self, disc_fake_hat, truePVfree, fakePVfree):
        gen_adv_loss = self.adv_loss(disc_fake_hat, torch.ones_like(disc_fake_hat))
        gen_rec_loss = self.recon_loss(truePVfree, fakePVfree)
        gen_loss = gen_adv_loss + self.lambda_recon * gen_rec_loss
        return gen_loss



class UNetLosses:
    def __init__(self, losses_params):
        if type(losses_params['recon_loss'])==list:
            self.recon_loss, self.lambdas  = [], []
            for (loss,lbda) in zip(losses_params['recon_loss'], losses_params['lambda_losses']):
                self.recon_loss.append(get_nn_loss(loss))
                self.lambdas.append(lbda)
        else:
            self.recon_loss = [get_nn_loss(loss_name=losses_params['recon_loss'])]
            self.lambdas = [1]

    def get_unet_loss(self, target, output):
        unet_loss = sum([lbda * loss(target,output) for (loss,lbda) in zip(self.recon_loss,self.lambdas)])
        return unet_loss