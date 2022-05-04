import torch
import matplotlib.pyplot as plt

def show_tensor_images(images):
    images_unflat = images.detach().cpu()
    batch_size = images_unflat.shape[0]

    fig,ax = plt.subplots(3,batch_size, squeeze=False)

    for k in range(batch_size):
        maxc = torch.max(images_unflat[k,0:2,:,:])
        minc = torch.min(images_unflat[k,0:2,:,:])
        ax[0,k].imshow(images_unflat[k,0,:,:], vmin = minc, vmax= maxc)
        ax[0,k].set_title('PVE')
        ax[1,k].imshow(images_unflat[k,1,:,:], vmin = minc, vmax= maxc)
        ax[1, k].set_title('PVfree')
        ax[2,k].imshow(images_unflat[k,2,:,:])
        ax[2, k].set_title('fakePVfree')
    plt.show()


def plot_losses(discriminator_losses,generator_losses):
    fig,ax1 = plt.subplots()


    p1 = ax1.plot(generator_losses, color = 'orange', label = 'Generator Loss')
    ax1.set_ylabel("Generator Loss", color = p1[0].get_color(), fontsize = 14)
    ax1.legend(loc=2) #upper left

    ax2 = ax1.twinx()
    p2 = ax2.plot(discriminator_losses,color = 'blue', label= 'Discriminator Loss')
    ax2.set_ylabel("Discriminator Loss", color = p1[0].get_color(),fontsize=14)
    ax2.legend(loc=1) #upper right

    ax2.set_xlabel('Iterations')
    ax2.set_title('Losses')

    plt.show()