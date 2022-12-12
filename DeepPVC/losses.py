import torch
from torch import nn

class PoissonLikelihood_loss(nn.Module):
    def __init__(self):
        super(PoissonLikelihood_loss, self).__init__()

    def forward(self, y_true,y_pred):
        eps = 1e-6
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_true = y_true.view(y_true.shape[0], -1)

        p_l = -y_true+y_pred*torch.log(y_true + eps)-torch.lgamma(y_pred+1)
        return -torch.mean(p_l)

class Pix2PixLosses:
    def __init__(self, losses_params):
        if losses_params['adv_loss']=='BCE':
            self.adv_loss = nn.BCEWithLogitsLoss()

        if losses_params['recon_loss']=='L1':
            self.recon_loss = nn.L1Loss()
        elif losses_params['recon_loss']=='L2':
            self.recon_loss = nn.MSELoss()

        self.lambda_recon = losses_params['lambda_recon']

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
        if losses_params['recon_loss']=='L1':
            self.recon_loss = [nn.L1Loss()]
        elif losses_params['recon_loss']=='L2':
            self.recon_loss = [nn.MSELoss()]
        elif type(losses_params['recon_loss'])==list:
            self.recon_loss = []
            for loss in losses_params['recon_loss']:
                if loss == "L1":
                    self.recon_loss.append(nn.L1Loss())
                elif loss == "L2":
                    self.recon_loss.append(nn.MSELoss())
                elif loss=="Poisson":
                    self.recon_loss.append(PoissonLikelihood_loss())

    def get_unet_loss(self, target, output):
        unet_loss = sum([loss(target,output) for loss in self.recon_loss])
        return unet_loss