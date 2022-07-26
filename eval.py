import torch
import numpy as np
import click
import glob
import random

from DeepPVC import plots, helpers_data,helpers_params, helpers, Pix2PixModel

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth', multiple = True) # 'path/to/saved/model.pth'
@click.option('--input', '-i', multiple = True)
@click.option('-n', help = 'If no input is specified, choose the number of random images on which you want to test')
@click.option('--dataset', help = 'path to the dataset folder in which to randomly select n images')
@click.option('--ref/--no-ref', default = True)
@click.option('--save', is_flag=True, help = "Wheter or not to save the corrected image")
@click.option('--mse', is_flag=True, help="Compute the MSE on the provided dataset")
def eval_click(pth, input, n, dataset, ref, save, mse):
    eval(pth, input, n, dataset, ref, save, mse)





def eval(pth, input,n,dataset,ref, save, mse):
    """ Evaluate visually a trained Pix2Pix (pth) on a given projection \n
        Output is the corrected projection

        Warning : the image will be normalized before PVC

    """
    print('Evaluation of the model on an image')

    if input:
        list_of_images = list(input)
        do_mse = False
    elif n:
        n = int(n)
        list_of_all_images = glob.glob(f'{dataset}/?????_PVE.mhd')
        Nimages = len(list_of_all_images)
        list_of_all_images = [list_of_all_images[i][:-8] for i in range(Nimages)]
        list_of_images = random.sample(list_of_all_images, n)
        if mse:
            do_mse = True
        else:
            do_mse = False

    else:
        do_mse = None
        print('ERROR : no input nor n specified. You need to specify EITHER a --input /path/to/input OR a number -n 10 of image to select randomly in the dataset')
        exit(0)

    device = helpers.get_auto_device("cpu")



    for one_pth in pth:

        pth_file = torch.load(one_pth, map_location=device)
        params = pth_file['params']
        helpers_params.check_params(params)

        norm = params['norm']
        print(norm)
        normalisation = params['data_normalisation']

        model = Pix2PixModel.PVEPix2PixModel(params=params, is_resume=False)
        model.load_model(one_pth)
        model.switch_device("cpu")
        model.switch_eval()

        model.plot_losses(save, wait = False, title = one_pth)
        print(model.params['MSE'])
        if do_mse:
            MSE = 0
            with torch.no_grad():
                for test_data in list_of_all_images:
                    input_array = helpers_data.load_image(test_data, True)
                    normalized_input_tensor = helpers_data.normalize(dataset_or_img=input_array, normtype=normalisation,norm=norm, to_torch=True, device='cpu')
                    tensor_PVE = normalized_input_tensor[:, 0,:, :, :]
                    output_tensor = model.Generator(tensor_PVE)

                    denormalized_output_array = helpers_data.denormalize(dataset_or_img=output_tensor, normtype=normalisation,norm=norm, to_numpy=True)

                    projPVfree = input_array[:,1,:,:,:]
                    projDeepPVC = denormalized_output_array
                    MSE += (np.mean((projDeepPVC - projPVfree) ** 2)) / Nimages
            print(f'MSE on the test dataset {dataset}:'+ "{:.3e}".format(MSE))

            #FIXME
            # if 'MSE' in model.params:
            #     done = False
            #     for n,ds_mse in enumerate(model.params['MSE']):
            #         if ds_mse[n][0]==dataset:
            #             ds_mse[n][1] = MSE
            #             done = True
            #     if not done:
            #         model.params['MSE'].append([dataset, MSE])
            # else:
            model.params['MSE'] = [[dataset,MSE]]
            model.save_model(output_path=one_pth, save_json=True)
            print('*' * 80)

        else:
            print(f'No calculation of MSE as no dataset is provided')

        model.show_infos()

        for input in list_of_images:
            is_ref = ref
            with torch.no_grad():
                input_array = helpers_data.load_image(input, is_ref)
                normalized_input_tensor = helpers_data.normalize(dataset_or_img = input_array,normtype=normalisation,norm = norm, to_torch=True, device='cpu')

                tensor_PVE = normalized_input_tensor[:,0,:,:,:]
                output_tensor = model.Generator(tensor_PVE)

                denormalized_output_array = helpers_data.denormalize(dataset_or_img = output_tensor,normtype=normalisation,norm=norm, to_numpy=True)


            imgs = np.concatenate((input_array[0,:,:,:,:],denormalized_output_array), axis=0)
            plots.show_images_profiles(imgs, profile=True, save = save, is_tensor=False, title = input)





if __name__ == '__main__':
    eval_click()
