import torch
import matplotlib.pyplot as plt
from train import test_dataloader,train_dataloader,training_params,losses_params
from models.Pix2PixModel import PVEPix2PixModel
from utils import plots
import numpy as np

pth_file = f'pix2pix_with_params_50.pth'



pix2pix = PVEPix2PixModel(load_pth=pth_file)
pix2pix.swith_eval()

pix2pix.plot_losses()


MSE = 0

# for (batch,test_data) in enumerate(test_dataloader):
#     if batch ==0:
#         truePVE = test_data[:,0,:,:].unsqueeze(1)
#         truePVfree = test_data[:,1,:,:].unsqueeze(1)
#
#         fakePVfree = pix2pix.Generator(truePVE.float()).detach()
#
#         plots.show_tensor_images(torch.cat((truePVE, truePVfree, fakePVfree), 1))
#
#         # np_PVE = truePVE[0,0,:,:].numpy()
#         # np_PVfree = truePVfree[0,0,:,:].numpy()
#         # np_fakePVfreee = fakePVfree[0,0,:,:].numpy()
#
#
#
#         # maxc = max(np.max(np_PVE), np.max(np_PVfree))
#
#         # center_indexes = np.where(np_PVfree == np.amax(np_PVfree))
#         # center_i = (np.mean(center_indexes[0])).astype(int)
#         # center_j = (np.mean(center_indexes[1])).astype(int)
#         #
#         # print(f'center_i : {center_i}')
#         # print(f'center_j : {center_j}')
#         #
#         # fig, ax = plt.subplots(2, 2, figsize=(8, 8))
#         # ax[0, 0].imshow(np_PVfree, cmap='Greys', vmin=0, vmax=maxc)
#         # ax[0, 0].hlines(center_i, 0, 127, color='blue', linestyle='dashed')
#         # ax[0, 0].vlines(center_j, 0, 127, color='orange', linestyle='dashed')
#         # ax[0, 0].set_title(f'PVfree projection')
#         # ax[0, 1].imshow(np_PVE, cmap='Greys', vmin=0, vmax=maxc)
#         # ax[0, 1].hlines(center_i, 0, 127, color='blue')
#         # ax[0, 1].vlines(center_j, 0, 127, color='orange')
#         # ax[0, 1].set_title(f'PVE projection')
#         # ax[0, 1].legend()
#         #
#         # ax[1, 0].plot(np_PVfree[center_i, :], label='PVfree', color='blue', linestyle='dashed')
#         # ax[1, 0].plot(np_PVE[center_i, :], label='PVE', color='blue')
#         # ax[1, 0].plot(maxc * (np_PVfree[center_i, :] > 0), color='black', linewidth=1)
#         #
#         # ax[1, 1].plot(np_PVfree[:, center_j], label='PVfree', color='orange', linestyle='dashed')
#         # ax[1, 1].plot(np_PVE[:, center_j], label='PVE', color='orange')
#         # ax[1, 1].plot(maxc * (np_PVfree[:, center_j] > 0), color='black', linewidth=1)
#         # ax[1, 1].legend()
#
#         plt.show()



