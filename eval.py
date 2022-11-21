import matplotlib.pyplot as plt
import torch
import numpy as np
import click
import random

from DeepPVC import plots, helpers_data,helpers_params, helpers, Models,dataset

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth')
@click.option('--input', '-i')
@click.option('-n',type=int, help = 'If no input is specified, choose the number of random images on which you want to test', default = 1)
@click.option('--dataset','dataset_path', help = 'path to the dataset folder in which to randomly select n images')
@click.option('--type', default = 'mhd', help = "mhd or mha", show_default = True)
@click.option('--ref/--no-ref')
@click.option('--save', is_flag=True, help = "Wheter or not to save the corrected image")
@click.option('--mse', is_flag=True, help="Compute the MSE on the provided dataset")
@click.option('--plot', is_flag=True)
def eval_click(pth, input, n, dataset_path,type, ref, save, mse, plot):
    eval(pth, input, n, dataset_path,type, ref, save, mse, plot)



def eval(pth, input,n,dataset_path,type,ref, save, mse, plot):
    device = helpers.get_auto_device("cpu")
    pth_file = torch.load(pth, map_location=device)
    print(pth_file['saving_date'])

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

        test_dataset = dataset.load_test_data(datatype=type,params=params,from_folder=dataset_path)
    else:
        print('ERROR : no input nor dataset specified. You need to specify EITHER a --input /path/to/input OR a number -n 10 of image to select randomly in the dataset')
        exit(0)

    print(test_dataset.shape)

    normalized_test_dataset = helpers_data.normalize(dataset_or_img=test_dataset,normtype=normalisation, norm = norm, to_torch=True, device=device)


    model.plot_losses(save, wait=False, title=pth)

    normalized_dataset_input = normalized_test_dataset[:, 0, :, :, :]

    if mse:

        with torch.no_grad():
            normalized_dataset_output = model.forward(normalized_dataset_input)
            denormalized_dataset_output = helpers_data.denormalize(dataset_or_img=normalized_dataset_output,normtype=normalisation,norm=norm,to_numpy=True)
            print(denormalized_dataset_output.shape)
            MSE = np.mean((test_dataset[:,-1,0,:,:] - denormalized_dataset_output[:,0,:,:])**2)
            print(f'MSE : '+ "{:.3e}".format(MSE))

            if 'MSE' in model.params:
                done = False
                for nnn,ds_mse in enumerate(model.params['MSE']):
                    if ds_mse[0]==dataset_path:
                        ds_mse[1] = MSE
                        done = True
                if not done:
                    model.params['MSE'].append([dataset_path, MSE])
            else:
                model.params['MSE'] = [[dataset_path,MSE]]

            model.save_model(output_path=pth, save_json=True)
            print('*' * 80)

        model.show_infos()

    if plot:
        N_data = test_dataset.shape[0]

        random_data_index = [random.randint(0,N_data-1) for _ in range(n)]

        print(random_data_index)

        for index in random_data_index:
            input_i = normalized_dataset_input[index,:,:,:][None,:,:,:]
            with torch.no_grad():
                output_i = model.forward(input_i)
                denormalized_output_i = helpers_data.denormalize(dataset_or_img=output_i,normtype=normalisation, norm=norm,to_numpy=True)


            fig,ax = plt.subplots(1,4)

            ax[0].imshow(test_dataset[index,0,0,:,:])
            ax[1].imshow(test_dataset[index,1,0,:,:])
            ax[2].imshow(test_dataset[index,2,0,:,:])
            ax[3].imshow(denormalized_output_i[0,0,:,:])
            plt.show()



if __name__ == '__main__':
    eval_click()
