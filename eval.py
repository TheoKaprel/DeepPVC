import torch
import numpy as np
import click
import glob
import random

from DeepPVC import plots, helpers_data,helpers_params, helpers, Models,dataset

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth')
@click.option('--input', '-i')
@click.option('-n', help = 'If no input is specified, choose the number of random images on which you want to test')
@click.option('--dataset','dataset_path', help = 'path to the dataset folder in which to randomly select n images')
@click.option('--type', default = 'mhd', help = "mhd or mha", show_default = True)
@click.option('--ref/--no-ref')
@click.option('--save', is_flag=True, help = "Wheter or not to save the corrected image")
@click.option('--mse', is_flag=True, help="Compute the MSE on the provided dataset")
def eval_click(pth, input, n, dataset_path,type, ref, save, mse):
    eval(pth, input, n, dataset_path,type, ref, save, mse)



def eval(pth, input,n,dataset_path,type,ref, save, mse):

    device = helpers.get_auto_device("cpu")
    pth_file = torch.load(pth, map_location=device)
    params = pth_file['params']
    helpers_params.check_params(params)
    network_architecture = params['network']
    norm = params['norm']
    print(norm)
    normalisation = params['data_normalisation']
    model = Models.ModelInstance(params=params, from_pth=pth)
    model.switch_device(device)
    model.switch_eval()

    print(input)
    if input:
        test_dataset = dataset.load_test_data(datatype=type,params=params,from_file=input,is_ref=ref)
    elif dataset_path:
        n = int(n)
        test_dataset = dataset.load_test_data(datatype=type,params=params,from_folder=dataset_path)
    else:
        print('ERROR : no input nor dataset specified. You need to specify EITHER a --input /path/to/input OR a number -n 10 of image to select randomly in the dataset')
        exit(0)

    print(test_dataset.shape)

    normalized_test_dataset = helpers_data.normalize(dataset_or_img=test_dataset,normtype=normalisation, norm = norm, to_torch=True, device=device)


    model.plot_losses(save, wait=False, title=pth)



    if mse:
        normalized_dataset_input = normalized_test_dataset[:,0,:,:,:]
        with torch.no_grad():
            normalized_dataset_output = model.forward(normalized_dataset_input)
            denormalized_dataset_output = helpers_data.denormalize(dataset_or_img=normalized_dataset_output,normtype=normalisation,norm=norm,to_numpy=True)
            print(denormalized_dataset_output.shape)
            MSE = np.mean((test_dataset[:,2,0,:,:] - denormalized_dataset_output[:,0,:,:])**2)
            print(f'MSE : '+ "{:.3e}".format(MSE))

            if 'MSE' in model.params:
                done = False
                for n,ds_mse in enumerate(model.params['MSE']):
                    if ds_mse[0]==dataset:
                        ds_mse[1] = MSE
                        done = True
                if not done:
                    model.params['MSE'].append([dataset, MSE])
            else:
                model.params['MSE'] = [[dataset,MSE]]

            model.save_model(output_path=pth, save_json=True)
            print('*' * 80)


        model.show_infos()

    #     for input in list_of_images:
    #         is_ref = ref
    #         with torch.no_grad():
    #             print(input)
    #             input_array = helpers_data.load_image(input, is_ref, type, noisy= network_architecture=='denoiser_pvc')
    #             normalized_input_tensor = helpers_data.normalize(dataset_or_img = input_array,normtype=normalisation,norm = norm, to_torch=True, device='cpu')
    #
    #             output_tensor = model.forward(normalized_input_tensor)
    #
    #             denormalized_output_array = helpers_data.denormalize(dataset_or_img = output_tensor,normtype=normalisation,norm=norm, to_numpy=True)
    #
    #
    #         # imgs = np.concatenate((input_array[0,:,:,:,:],denormalized_output_array), axis=0)
    #         # # plots.show_images_profiles_denoiser_pvc(imgs, title = input)
    #         # plots.show_images_profiles(imgs, profile=True, noisy=network_architecture=='denoiser_pvc', save=False,is_tensor=False,title=input)
    #
    #



if __name__ == '__main__':
    eval_click()
