import torch
from torch import nn

class Pix2PixLosses:
    def __init__(self, losses_params):
        if losses_params['adv_loss']=='BCE':
            self.adv_loss = nn.BCEWithLogitsLoss()

        if losses_params['recon_loss']=='L1':
            self.recon_loss = nn.L1Loss()

        self.lambda_recon = losses_params['lambda_recon']

    def get_adv_loss(self):
        return self.adv_loss
    def get_recon_loss(self):
        return self.recon_loss

    def get_gen_loss(self,gen, disc, projPVfree, projPVE):
        '''
        Return the loss of the generator given inputs.
        Parameters:
            gen: the generator; takes the condition and returns potential images
            disc: the discriminator; takes images and the condition and
              returns real/fake prediction matrices
            real/projPVfree: the real images (e.g. maps) to be used to evaluate the reconstruction
            condition/projPVE: the source images (e.g. satellite imagery) which are used to produce the real images
            adv_criterion: the adversarial loss function; takes the discriminator
                      predictions and the true labels and returns a adversarial
                      loss (which you aim to minimize)
            recon_criterion: the reconstruction loss function; takes the generator
                        outputs and the real images and returns a reconstructuion
                        loss (which you aim to minimize)
            lambda_recon: the degree to which the reconstruction loss should be weighted in the sum
        '''
        # Steps: 1) Generate the fake images, based on the conditions.
        #        2) Evaluate the fake images and the condition with the discriminator.
        #        3) Calculate the adversarial and reconstruction losses.
        #        4) Add the two losses, weighting the reconstruction loss appropriately.

        fakePVfree = gen(projPVE.float())
        disc_fake_hat = disc(fakePVfree.float(), projPVE.float())
        gen_adv_loss = self.adv_loss(disc_fake_hat, torch.ones_like(disc_fake_hat))
        gen_rec_loss = self.recon_loss(projPVfree, fakePVfree)
        gen_loss = gen_adv_loss + self.lambda_recon * gen_rec_loss
        return gen_loss