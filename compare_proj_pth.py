import torch
import matplotlib.pyplot as plt
import numpy as np
import click
import itk
import glob
import random

from DeepPVC import plots, helpers_data, helpers, Pix2PixModel

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth', multiple = True) # 'path/to/saved/model.pth'
@click.option('--input', '-i', multiple = True)
@click.option('-n', help = 'If no input is specified, choose the number of random images on which you want to test')
@click.option('--dataset', help = 'path to the dataset folder in which to randomly select n images')
@click.option('--ref/--no-ref', default = True)
def compare_proj_pth_click(pth, input, n, dataset, ref):
    compare_proj_pth(pth, input, n, dataset, ref)





def compare_proj_pth(pth, input,n,dataset,ref):

    if input:
        list_of_images = list(input)
    elif (n and dataset):
        n = int(n)
        list_of_all_images = glob.glob(f'{dataset}/?????.mhd')
        Nimages = len(list_of_all_images)
        list_index = [random.randint(0,Nimages) for _ in range(n)]
        list_of_images = [list_of_all_images[list_index[i]][:-4] for i in range(len(list_index))]
    else:
        print('ERROR : no input nor n/dataset specified. You need to specify EITHER a --input /path/to/input OR a number -n 10 of image and a --dataset /pth/to/dataset to select randomly in the dataset')
        list_of_images = []
        exit(0)

    device = helpers.get_auto_device("cpu")

    nImgs = len(list_of_images)
    nPth = len(pth)

    if nPth<2:
        print('ERROR : the number of pth file should be > 1 since this code is made to compare 2 or more pth on projections')
        exit(0)


    for img in list_of_images:
        is_ref = ref
        input_array = helpers_data.load_image(img, is_ref)

        projs_DeepPVC = np.zeros((nPth,128,128))

        for idpth in range(nPth):
            one_pth = pth[idpth]
            pth_file = torch.load(one_pth, map_location=device)
            params = pth_file['params']
            norm = params['norm']
            normalisation = params['data_normalisation']
            model = Pix2PixModel.PVEPix2PixModel(params=params, is_resume=False)
            model.load_model(one_pth)
            model.switch_device("cpu")
            model.switch_eval()
            model.show_infos()

            normalized_input_tensor = helpers_data.normalize(dataset_or_img=input_array, normtype=normalisation,
                                                             norm=norm,
                                                             to_torch=True, device='cpu')
            tensor_PVE = normalized_input_tensor[:, 0, :, :][:, None, :, :]

            output_tensor = model.test(tensor_PVE)

            denormalized_output_array = helpers_data.denormalize(dataset_or_img = output_tensor,normtype=normalisation,norm=norm, to_numpy=True)


            projs_DeepPVC[idpth] = np.squeeze(denormalized_output_array)


        fig, ax = plt.subplots(2,nPth)
        input_array_sq = np.squeeze(input_array)
        vmin = min((np.min(input_array_sq), np.min(projs_DeepPVC)))
        vmax = max((np.max(input_array_sq), np.max(projs_DeepPVC)))

        ax[0,0].imshow(input_array_sq[0,:,:], vmin=vmin, vmax=vmax)
        ax[0,0].set_title('PVE')

        ax[0,1].imshow(input_array_sq[1,:,:], vmin=vmin, vmax=vmax)
        ax[0,1].set_title('PVfree')

        for k in range(nPth):
            ax[1,k].imshow(projs_DeepPVC[k,:,:], vmin=vmin, vmax=vmax)
            # ax[1,k].imshow((projs_DeepPVC[k,:,:] - input_array_sq[1,:,:])**2)
            ax[1,k].set_title(pth[k])
            mse = np.mean((projs_DeepPVC[k,:,:] - input_array_sq[1,:,:])**2)
            ax[1,k].set_ylabel("MSE = {}".format(mse))
        plt.suptitle(img)
        plt.show()






if __name__ == '__main__':
    compare_proj_pth_click()
