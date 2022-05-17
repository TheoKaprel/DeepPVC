import torch
import matplotlib.pyplot as plt
from utils import plots,helpers_data
import numpy as np
import click
from models.Pix2PixModel import PVEPix2PixModel
import itk



CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('pth')
@click.argument('input')
@click.option('--save', is_flag=True, help = "Wheter or not to save the corrected image")
@click.option('--output', '-o', help = 'Output filename (.mhd)')
def eval_one_image(pth, input, save, output):
    """ Evaluate visually a trained Pix2Pix (pth) on a given projection \n
        Output is the corrected projection

        PTH: path to the pth file containing the model and all its parameters (.pth)
        INPUT: path to the input projection to correct (.mhd)

        Warning : the image will be normalized before PVC

    """
    print('Evaluation of the model on an image')

    pth_file = torch.load(pth)
    params = pth_file['params']
    model = PVEPix2PixModel(params=params)
    model.load_model(pth)


    model.switch_device("cpu")
    model.switch_eval()

    model.show_infos()

    model.plot_losses()

    # input_tensor = helpers_data.load_tensor_from_mhd(input)

    input_tensor = helpers_data.load_tensor_PVE_PVfree_from_mhd(input)
    print(input_tensor.shape)
    tensor_PVE = input_tensor[:,0,:,:]
    tensor_PVE = tensor_PVE[:,None,:,:]


    output_tensor = model.test(tensor_PVE)
    imgs = torch.cat((input_tensor,output_tensor), dim=1)
    print(imgs.shape)

    # plots.show_two_images(input_tensor, output_tensor)

    plots.show_tensor_images(imgs)




if __name__ == '__main__':
    eval_one_image()