import matplotlib.pyplot as plt
import torch
import numpy as np
import click
import random

from DeepPVC import plots, helpers_data,helpers_params, helpers, Models,dataset

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth', multiple = True)
@click.option('--input', '-i')
@click.option('-n',type=int, help = 'If no input is specified, choose the number of random images on which you want to test', default = 1)
@click.option('--dataset','dataset_path', help = 'path to the dataset folder in which to randomly select n images')
@click.option('--type', default = 'mhd', help = "mhd or mha", show_default = True)
@click.option('--ref/--no-ref')
@click.option('--mse', is_flag=True, help="Compute the MSE on the provided dataset")
@click.option('--plot', is_flag=True)
@click.option('-v', '--verbose', count=True)
def eval_click(pth, input, n, dataset_path,type, ref, mse, plot, verbose):

    if mse:
        eval_mse(pth, input, n, dataset_path,type, ref, verbose)
    if plot:
        eval_plot(pth, input, n, dataset_path,type, ref, verbose)



def eval_mse(lpth, input,n,dataset_path,type,ref, verbose):
    device = helpers.get_auto_device("cpu")

    dict_mse = {}

    for pth in lpth:
        pth_file = torch.load(pth, map_location=device)

        params = pth_file['params']
        helpers_params.check_params(params)
        pth_ref = params['ref']
        norm = params['norm']

        normalisation = params['data_normalisation']
        model = Models.ModelInstance(params=params, from_pth=pth)
        model.switch_device(device)
        model.switch_eval()

        if input:
            test_dataset = dataset.load_test_data(datatype=type,params=params,from_file=input,is_ref=ref)
        elif dataset_path:
            test_dataset = dataset.load_test_data(datatype=type,params=params,from_folder=dataset_path)
        else:
            print('ERROR : no input nor dataset specified. You need to specify EITHER a --input /path/to/input OR a number -n 10 of image to select randomly in the dataset')
            exit(0)


        normalized_test_dataset = helpers_data.normalize(dataset_or_img=test_dataset,normtype=normalisation, norm = norm, to_torch=True, device=device)

        if verbose>1:
            model.plot_losses(save=False, wait=False, title=pth)

        normalized_dataset_input = normalized_test_dataset[:, 0, :, :, :]

        with torch.no_grad():
            normalized_dataset_output = model.forward(normalized_dataset_input)
            denormalized_dataset_output = helpers_data.denormalize(dataset_or_img=normalized_dataset_output,normtype=normalisation,norm=norm,to_numpy=True)
            MSE = np.mean((test_dataset[:,-1,0,:,:] - denormalized_dataset_output[:,0,:,:])**2)
            print(f'MSE : '+ "{:.3e}".format(MSE))

            dict_mse[pth_ref] = MSE

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
            if verbose > 0:
                model.show_infos()

            print('*' * 80)



    fig,ax = plt.subplots()
    plt.bar(range(len(dict_mse)), list(dict_mse.values()), align='center')
    plt.xticks(range(len(dict_mse)), list(dict_mse.keys()))


    plt.show()



def eval_plot(lpth, input, n, dataset_path, type, ref, verbose):
    device = helpers.get_auto_device("cpu")

    random_data_index = []
    first = True

    dict_data = {}

    for pth in lpth:
        pth_file = torch.load(pth, map_location=device)

        params = pth_file['params']
        helpers_params.check_params(params)
        pth_ref = params['ref']
        norm = params['norm']

        normalisation = params['data_normalisation']
        model = Models.ModelInstance(params=params, from_pth=pth)
        model.switch_device(device)
        model.switch_eval()

        if verbose>0:
            model.show_infos()

        if input:
            test_dataset = dataset.load_test_data(datatype=type,params=params,from_file=input,is_ref=ref)
        elif dataset_path:
            test_dataset = dataset.load_test_data(datatype=type,params=params,from_folder=dataset_path)
        else:
            print('ERROR : no input nor dataset specified. You need to specify EITHER a --input /path/to/input OR a number -n 10 of image to select randomly in the dataset')
            exit(0)

        if first:
            N_data = test_dataset.shape[0]
            random_data_index = [random.randint(0, N_data - 1) for _ in range(n)]
            for id in random_data_index:
                dict_data[id] = {}
                dict_data[id]['PVE_noisy'] = test_dataset[id, 0, 0, :, :]
                dict_data[id]['PVE'] = test_dataset[id, 1, 0, :, :]
                dict_data[id]['noPVE'] = test_dataset[id, 2, 0, :, :]

            first=False

        normalized_test_dataset = helpers_data.normalize(dataset_or_img=test_dataset,normtype=normalisation, norm = norm, to_torch=True, device=device)

        normalized_dataset_input = normalized_test_dataset[:, 0, :, :, :]


        for index in random_data_index:
            input_i = normalized_dataset_input[index, :, :, :][None, :, :, :]
            with torch.no_grad():
                output_i = model.forward(input_i)
                denormalized_output_i = helpers_data.denormalize(dataset_or_img=output_i,
                                                                 normtype=normalisation, norm=norm,
                                                                 to_numpy=True)
                dict_data[index][pth_ref] = denormalized_output_i[0, 0, :, :]





    for id in random_data_index:
        fig, axs = plt.subplots(2, max(3, len(lpth)))

        dict_data_id = dict_data[id]
        vmin = 0
        vmax = max([np.max(img_idk) for img_idk in dict_data_id.values()])

        for key,ax in zip(dict_data_id,axs.ravel()):
            ax.imshow(dict_data_id[key], vmin = vmin, vmax = vmax)
            ax.set_title(key)

        plt.show()


if __name__ == '__main__':
    eval_click()
