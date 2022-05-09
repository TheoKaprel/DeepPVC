import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn,optim
from torchvision.utils import make_grid
from data.dataset import load_data
from options import base_options
from models import networks
from models.Pix2PixModel import  PVEPix2PixModel
from utils import plots


data_params = {'dataset_path': '../PVE_data/Analytical_data/dataset','training_batchsize':5, 'test_batchsize':5, 'training_prct':0.8}
training_params = {'n_epochs':10,'learning_rate':0.0002, 'input_channels':1, 'hidden_channels_gen':64, 'hidden_channels_disc':9, 'display_step':40,'generator_optimizer':'Adam','training_device':'cpu'}
losses = {'adv_loss':nn.BCEWithLogitsLoss(), 'recon_loss':nn.L1Loss(),'lambda_recon':200 }


train_dataloader,test_dataloader = load_data(dataset_path=data_params['dataset_path'],
                                             training_batchsize=data_params['training_batchsize'], testing_batchsize=data_params['test_batchsize'],prct_train=data_params['training_prct'])



DeepPVEModel = PVEPix2PixModel(training_params=training_params)


print(DeepPVEModel)


#
# generator_losses = []
# discriminator_losses = []
#
# def train():
#     mean_generator_loss = 0
#     mean_discriminator_loss = 0
#
#
#     for epoch in range(1,n_epochs+1):
#         print(f'Epoch {epoch}/{n_epochs}')
#         for step,batch in enumerate(train_dataloader):
#             step = step+1
#             print(f'step {step}/{len(train_dataloader)}..............')
#             truePVE = batch[:,0,:, :].unsqueeze(1).to(device)
#             truePVfree = batch[:,1,:,:].unsqueeze(1).to(device)
#
#             ## Update Discriminator
#             disc_opt.zero_grad()
#             with torch.no_grad():
#                 fakePVfree = Gen(truePVE.float())
#
#             disc_fake_hat = Disc(fakePVfree.detach().float(),truePVE.float())
#             disc_fake_loss = adv_criterion(disc_fake_hat,torch.zeros_like(disc_fake_hat))
#
#             disc_real_hat = Disc(truePVfree.float(),truePVE.float())
#             disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
#
#             disc_loss = (disc_fake_loss+disc_real_loss)/2
#             disc_loss.backward(retain_graph = True)
#             disc_opt.step()
#
#             ## Update Generator
#             gen_opt.zero_grad()
#             gen_loss = get_gen_loss(Gen, Disc, truePVfree, truePVE, adv_criterion, recon_criterion, lambda_recon)
#             gen_loss.backward()
#             gen_opt.step()
#
#             # Keep track of the average discriminator loss
#             mean_discriminator_loss += disc_loss.item() / display_step
#             # Keep track of the average generator loss
#             mean_generator_loss += gen_loss.item() / display_step
#
#             discriminator_losses.append(disc_loss.item())
#             generator_losses.append(gen_loss.item())
#
#
#             ### Visualization code ###
#             if step % display_step == 0:
#
#                 print(f"Epoch {epoch}: Step {step}: Generator loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
#
#                 plots.show_tensor_images(torch.cat((truePVE,truePVfree,fakePVfree), 1))
#
#                 mean_generator_loss = 0
#                 mean_discriminator_loss = 0
#
#
#     torch.save({'epoch':n_epochs,
#                 'gen': Gen.state_dict(),
#                 'gen_opt': gen_opt.state_dict(),
#                 'disc': Disc.state_dict(),
#                 'disc_opt': disc_opt.state_dict(),
#                 'gen_losses':generator_losses,
#                 'disc_losses':discriminator_losses
#                 }, f"pix2pix_{n_epochs}.pth")
#
#
#
# if __name__ == '__main__':
#     train()
#     plots.plot_losses(discriminator_losses,generator_losses)
#
