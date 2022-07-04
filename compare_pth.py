import torch
import matplotlib.pyplot as plt
import numpy as np
import click
import itk
import glob
import random
import json

from DeepPVC import plots, helpers_data, helpers, Pix2PixModel
from DeepPVC import dataset as ds

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth', multiple = True) # 'path/to/saved/model.pth'
@click.option('--proj', is_flag = True, default = False)
@click.option('--input', '-i', multiple = True)
@click.option('-n', help = 'If no input is specified, choose the number of random images on which you want to test')
@click.option('--dataset', help = 'path to the dataset folder in which to randomly select n images')
@click.option('--ref/--no-ref', default = True)
@click.option('--losses', is_flag = True, default = False)
@click.option('--mse', is_flag = True, default = False, help = 'Compute mse on --dataset')
def compare_proj_pth_click(pth,proj, input, n, dataset, ref, losses, mse):
    compare_proj_pth(pth,proj, input, n, dataset, ref, losses, mse)





def compare_proj_pth(pth,proj, input, n, dataset, ref, losses, mse):

    # load models
    device = helpers.get_auto_device("cpu")
    nPth = len(pth)
    if nPth<2:
        print('ERROR : the number of pth file should be > 1 since this code is meant to compare 2 or more pth on projections')
        exit(0)

    list_models = []
    for idpth in range(nPth):
        one_pth = pth[idpth]
        pth_file = torch.load(one_pth, map_location=device)
        params = pth_file['params']
        model = Pix2PixModel.PVEPix2PixModel(params=params, is_resume=False)
        model.load_model(one_pth)
        model.switch_device("cpu")
        model.switch_eval()
        formatted_json = model.show_infos()
        list_models.append(model)
        if losses:
            model.plot_losses(save=False, wait = True, title = one_pth)
        # jsonString = json.dumps(model.params)
        jsonFile = open("test_json.json", "w")
        jsonFile.write(formatted_json)
        jsonFile.close()

    if losses:
        plt.show()

    if proj:
        # load projections
        if input:
            list_of_images = list(input)
        elif (n and dataset):
            n = int(n)
            list_of_all_images = glob.glob(f'{dataset}/?????.mhd')
            Nimages = len(list_of_all_images)
            list_index = [random.randint(0, Nimages) for _ in range(n)]
            list_of_images = [list_of_all_images[list_index[i]][:-4] for i in range(len(list_index))]
        else:
            print(
                'ERROR : no input nor n/dataset specified. You need to specify EITHER a --input /path/to/input OR a number -n 10 of image and a --dataset /pth/to/dataset to select randomly in the dataset')
            list_of_images = []
            exit(0)

    if mse:
        test_dataset = ds.construct_dataset_from_path(dataset_path=dataset)
        nb_testing_data = test_dataset.shape[0]
        print(f'nb_test : {nb_testing_data}')
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=list_models[0].params['test_batchsize'], shuffle=True)


        MSE = np.zeros(nPth)
        for test_it, batch in enumerate(test_dataloader):
            for m in range(nPth):
                model = list_models[m]
                model.input_data(batch)
                fakePVfree = model.test(model.truePVE)

                denormalized_input = helpers_data.denormalize(model.truePVfree,
                                                              normtype=model.params['data_normalisation'], norm=model.params['norm'],
                                                              to_numpy=True)
                denormalized_output = helpers_data.denormalize(fakePVfree, normtype=model.params['data_normalisation'],
                                                               norm=model.params['norm'], to_numpy=True)
                MSE[m]+= np.sum(np.mean((denormalized_output - denormalized_input) ** 2, axis=(2, 3))) / nb_testing_data
        print(MSE)

    if proj:
        for img in list_of_images:
            is_ref = ref
            input_array = helpers_data.load_image(img, is_ref)

            projs_DeepPVC = np.zeros((nPth,128,128))

            for idpth in range(nPth):
                model = list_models[idpth]
                normalisation = model.params['data_normalisation']
                norm = model.params['norm']

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
