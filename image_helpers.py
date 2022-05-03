import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import itk

def show_tensor_images(image_tensor, num_images=25):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    plots and prints the images in an uniform grid.
    '''
    image_shifted = image_tensor
    size = image_tensor.shape[1:]
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()



def open_image_to_tensor(list_images):


    img = itk.imread('../PVE_data/Analytical_data/dataset/WVQNF_PVE.mhd')
    img_np = itk.array_from_image(img)
    input = torch.from_numpy(img_np)