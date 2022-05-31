import torch
import matplotlib.pyplot as plt
from utils import plots,helpers_data
import numpy as np
import click
from models.Pix2PixModel import PVEPix2PixModel
import itk
import glob
import random






CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('pth')
@click.option('--input')
@click.option('-n', help = 'If no input is specified, choose the number of random images on which you want to test')
@click.option('--dataset', help = 'path to the dataset folder in which to randomly select n images')
@click.option('--ref/--no-ref', default = True)
@click.option('--save', is_flag=True, help = "Wheter or not to save the corrected image")
@click.option('--output', '-o', help = 'Output filename (.mhd)')

def eval_one_image(pth, input,n,dataset,ref, save, output):
    """ Evaluate visually a trained Pix2Pix (pth) on a given projection \n
        Output is the corrected projection

        PTH: path to the pth file containing the model and all its parameters (.pth)
        INPUT: path to the input projection to correct (.mhd)
        OR
        n

        Warning : the image will be normalized before PVC

    """
    print('Evaluation of the model on an image')

    if input:
        list_of_images = [input]
    elif n:
        n = int(n)
        list_of_all_images = glob.glob(f'{dataset}/?????.mhd')
        Nimages = len(list_of_all_images)
        list_index = [random.randint(0,Nimages) for _ in range(n)]
        list_of_images = [list_of_all_images[list_index[i]][:-4] for i in range(n)]

    else:
        print('ERROR : no input nor n specified. You need to specify EITHER a --input /path/to/input OR a number -n 10 of image to select randomly in the dataset')
        exit(0)


    pth_file = torch.load(pth)
    params = pth_file['params']
    data_normalisation = params['data_normalisation']
    model = PVEPix2PixModel(params=params, is_resume=False)
    model.load_model(pth)
    model.switch_device("cpu")
    model.switch_eval()
    model.show_infos()
    model.plot_losses()

    for input in list_of_images:
        is_ref = ref
        input_tensor,norms = helpers_data.load_image(input, data_normalisation, is_ref)

        tensor_PVE = input_tensor[:,0,:,:]
        tensor_PVE = tensor_PVE[:,None,:,:]
        output_tensor = model.test(tensor_PVE)

        imgs = torch.cat((input_tensor,output_tensor), dim=1)
        plots.show_images_profiles(imgs, profile=True)





if __name__ == '__main__':
    eval_one_image()