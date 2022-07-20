import torch
from torch import nn

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