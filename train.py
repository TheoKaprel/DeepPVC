import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn,optim
from torchvision.utils import make_grid
from data.dataset import load_data
from options import base_options
from models import networks
from utils import plots



dataset_path = '../PVE_data/Analytical_data/dataset'
batch_size = 5
prct_train = 0.8


adv_criterion = nn.BCEWithLogitsLoss()
recon_criterion = nn.L1Loss()
lambda_recon = 200

n_epochs = 1
input_channels = 1
hidden_channels_gen = 64
hidden_channels_disc = 9
display_step = 40

lr = 0.0002
device = 'cpu'




train_dataloader,test_dataloader = load_data(dataset_path=dataset_path,batchsize=batch_size,prct_train=prct_train)


Gen = networks.UNetGenerator(input_channel=input_channels, ngc=hidden_channels_gen, output_channel=input_channels)
Disc = networks.NLayerDiscriminator(input_channel=2*input_channels, ndc=hidden_channels_disc, output_channel=input_channels)


gen_opt = optim.Adam(Gen.parameters(), lr=lr)
disc_opt = optim.Adam(Disc.parameters(), lr=lr)


def get_gen_loss(gen, disc, projPVfree, projPVE, adv_criterion, recon_criterion, lambda_recon):
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
    gen_adv_loss = adv_criterion(disc_fake_hat, torch.ones_like(disc_fake_hat))
    gen_rec_loss = recon_criterion(projPVfree, fakePVfree)
    gen_loss = gen_adv_loss + lambda_recon * gen_rec_loss

    return gen_loss



generator_losses = []
discriminator_losses = []

def train():
    mean_generator_loss = 0
    mean_discriminator_loss = 0


    for epoch in range(1,n_epochs+1):
        print(f'Epoch {epoch}/{n_epochs}')
        for step,batch in enumerate(train_dataloader):
            step = step+1
            print(f'step {step}/{len(train_dataloader)}..............')
            truePVE = batch[:,0,:, :].unsqueeze(1).to(device)
            truePVfree = batch[:,1,:,:].unsqueeze(1).to(device)

            ## Update Discriminator
            disc_opt.zero_grad()
            with torch.no_grad():
                fakePVfree = Gen(truePVE.float())

            disc_fake_hat = Disc(fakePVfree.detach().float(),truePVE.float())
            disc_fake_loss = adv_criterion(disc_fake_hat,torch.zeros_like(disc_fake_hat))

            disc_real_hat = Disc(truePVfree.float(),truePVE.float())
            disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))

            disc_loss = (disc_fake_loss+disc_real_loss)/2
            disc_loss.backward(retain_graph = True)
            disc_opt.step()

            ## Update Generator
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(Gen, Disc, truePVfree, truePVE, adv_criterion, recon_criterion, lambda_recon)
            gen_loss.backward()
            gen_opt.step()

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            discriminator_losses.append(disc_loss.item())
            generator_losses.append(gen_loss.item())


            ### Visualization code ###
            if step % display_step == 0:

                print(f"Epoch {epoch}: Step {step}: Generator loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")

                plots.show_tensor_images(torch.cat((truePVE,truePVfree,fakePVfree), 1))

                mean_generator_loss = 0
                mean_discriminator_loss = 0


    torch.save({'epoch':n_epochs,
                'gen': Gen.state_dict(),
                'gen_opt': gen_opt.state_dict(),
                'disc': Disc.state_dict(),
                'disc_opt': disc_opt.state_dict(),
                'gen_losses':generator_losses,
                'disc_losses':discriminator_losses
                }, f"pix2pix_{n_epochs}.pth")



if __name__ == '__main__':
    train()
    plots.plot_losses(discriminator_losses,generator_losses)

