import matplotlib.pyplot as plt
import torch
import numpy as np
import click
import random

from DeepPVC import plots, helpers_data,helpers_params, helpers, Models,dataset, helpers_functions

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth', multiple = True)
@click.option('--input', '-i')
@click.option('-n',type=int, help = 'If no input is specified, choose the number of random images on which you want to test', default = 1)
@click.option('--dataset','dataset_path', help = 'path to the dataset folder in which to randomly select n images')
@click.option('--type', default = 'mhd', help = "mhd or mha", show_default = True)
@click.option('--ref/--no-ref')
@click.option('--error', is_flag=True, help="Compute the MSE on the provided dataset")
@click.option('--plot', is_flag=True)
@click.option('-v', '--verbose', count=True)
def eval_click(pth, input, n, dataset_path,type, ref, error, plot, verbose):

    if error:
        eval_error(pth, input, dataset_path,type, ref, verbose)
    if plot:
        eval_plot(pth, input, n, dataset_path,type, ref, verbose)


def add_or_modify_error(dataset_path, params, error_ref, error_val):
    if error_ref in params:
        done = False
        for nnn, ds_mse in enumerate(params[error_ref]):
            if ds_mse[0] == dataset_path:
                ds_mse[1] = error_val
                done = True
        if not done:
            params[error_ref].append([dataset_path, error_val])
    else:
        params[error_ref] = [[dataset_path, error_val]]

    return params


def eval_error(lpth, input,dataset_path,type,ref, verbose):
    device = helpers.get_auto_device("cpu")

    dict_mse,dict_std_mse = {},{}
    dict_mae,dict_std_mae = {},{}

    for pth in lpth:
        pth_file = torch.load(pth, map_location=device)

        params = pth_file['params']
        pth_ref = params['ref']

        model = Models.ModelInstance(params=params, from_pth=pth, resume_training=False)
        model.switch_device(device)
        model.switch_eval()

        if input:
            test_dataset = dataset.load_test_data(datatype=type,params=params,from_file=input,is_ref=ref)
        elif dataset_path:
            test_dataset = dataset.load_test_data(datatype=type,params=params,from_folder=dataset_path)
        else:
            print('ERROR : no input nor dataset specified. You need to specify EITHER a --input /path/to/input OR a number -n 10 of image to select randomly in the dataset')
            exit(0)


        with torch.no_grad():
            (MNRMSE,std_NRMSE), (MNMAE,std_NMAE) = helpers_functions.validation_errors(test_dataset_numpy=test_dataset, model=model, do_NRMSE=True, do_NMAE=True)
            print(f'Mean NRMSE : '+ "{:.3e}".format(MNRMSE) + "  (std={:.3e})".format(std_NRMSE))
            print(f'Mean MNMAE : '+ "{:.3e}".format(MNMAE)  + "  (std={:.3e})".format(std_NMAE))

            dict_mse[pth_ref] = MNRMSE
            dict_mae[pth_ref] = MNMAE
            dict_std_mse[pth_ref] = std_NRMSE
            dict_std_mae[pth_ref] = std_NMAE

            model.params = add_or_modify_error(dataset_path=dataset_path, params=model.params, error_ref='MNRMSE', error_val=MNRMSE)
            model.params = add_or_modify_error(dataset_path=dataset_path, params=model.params, error_ref='MNMAE', error_val=MNMAE)

            if verbose > 0:
                model.show_infos()

            print('*' * 80)



    fig,ax = plt.subplots(1,2)
    ax[0].bar(range(len(dict_mse)), list(dict_mse.values()),yerr=dict_std_mse.values(), align='center', capsize=5)
    ax[0].set_xticks(ticks = range(len(dict_mse)))
    ax[0].set_xticklabels(list(dict_mse.keys()))
    ax[0].set_title('Mean NRMSE')


    ax[1].bar(range(len(dict_mae)), list(dict_mae.values()),yerr=dict_std_mae.values(), align='center', capsize=5)
    ax[1].set_xticks(ticks = range(len(dict_mae)))
    ax[1].set_xticklabels(list(dict_mae.keys()))
    ax[1].set_title('Mean NMAE')


    plt.show()



def eval_plot(lpth, input, n, dataset_path, type, ref, verbose):
    device = helpers.get_auto_device("cpu")

    random_data_index = []

    dict_data = {}
    lpth_ref = []

    for (id,pth) in enumerate(lpth):
        pth_file = torch.load(pth, map_location=device)

        params = pth_file['params']
        pth_ref = params['ref']
        lpth_ref.append(pth_ref)

        normalisation = params['data_normalisation']
        model = Models.ModelInstance(params=params, from_pth=pth,resume_training=False)
        model.switch_device(device)
        model.switch_eval()

        if verbose>0:
            model.show_infos()
            if verbose > 1:
                model.plot_losses(save=False, wait=True, title=pth)

        if input:
            test_dataset = dataset.load_test_data(datatype=type,params=params,from_file=input,is_ref=ref)
        elif dataset_path:
            test_dataset = dataset.load_test_data(datatype=type,params=params,from_folder=dataset_path)
        else:
            print('ERROR : no input nor dataset specified. You need to specify EITHER a --input /path/to/input OR a number -n 10 of image to select randomly in the dataset')
            exit(0)

        if id==0:
            N_data = test_dataset.shape[0]
            random_data_index = [random.randint(0, N_data - 1) for _ in range(n)]
            for id in random_data_index:
                dict_data[id] = {}
                dict_data[id]['PVE_noisy'] = test_dataset[id, 0, 0, :, :]
                dict_data[id]['PVE'] = test_dataset[id, 1, 0, :, :]
                dict_data[id]['noPVE'] = test_dataset[id, 2, 0, :, :]


        for index in random_data_index:
            input_i = test_dataset[index,0, :, :, :][None,None, :, :, :]
            with torch.no_grad():
                norm_input_i = helpers_data.compute_norm_eval(dataset_or_img=input_i,data_normalisation=normalisation)
                normalized_input_i = helpers_data.normalize_eval(dataset_or_img=input_i,data_normalisation=normalisation,norm=norm_input_i,params=params,to_torch=True)

                output_i = model.forward(normalized_input_i)
                denormalized_output_i = helpers_data.denormalize_eval(dataset_or_img=output_i,data_normalisation=normalisation,norm=norm_input_i,params=params,to_numpy=True)
                dict_data[index][pth_ref] = [denormalized_output_i[0, 0, :, :]]

                if verbose>2:
                    denoisedPVE = model.denoisedPVE
                    denormalized_denoisedPVE = helpers_data.denormalize_eval(dataset_or_img=denoisedPVE,data_normalisation=normalisation,norm=norm_input_i,params=params,to_numpy=True)
                    dict_data[index][pth_ref].append(denormalized_denoisedPVE[0,0,:,:])


    if verbose>1:
        plt.show()



    for id in random_data_index:
        if verbose>2:
            n_rows = 3
        else:
            n_rows = 2

        fig, axs = plt.subplots(n_rows, max(3, len(lpth)),figsize=(20, 12))

        dict_data_id = dict_data[id]
        vmin = 0
        vmax = max([np.max(img_idk[0]) for img_idk in dict_data_id.values()])

        axs[0,0].imshow(dict_data_id['PVE_noisy'], vmin = vmin, vmax = vmax)
        axs[0,0].set_title('PVE_noisy')
        axs[0,1].imshow(dict_data_id['PVE'], vmin = vmin, vmax = vmax)
        axs[0,1].set_title('PVE')
        axs[0,2].imshow(dict_data_id['noPVE'], vmin = vmin, vmax = vmax)
        axs[0,2].set_title('noPVE')


        for i,(pth_ref,ax) in enumerate(zip(lpth_ref,axs[1,:])):

            ax.imshow(dict_data_id[pth_ref][0], vmin = vmin, vmax = vmax)
            ax.set_title(pth_ref)

            if i>=3:
                axs[0,i].axis('off')

        if verbose>2:
            for i, (pth_ref, ax) in enumerate(zip(lpth_ref, axs[2, :])):
                ax.imshow(dict_data_id[pth_ref][1], vmin=vmin, vmax=vmax)
                ax.set_title(pth_ref)

        if len(lpth_ref)<3:
            for j in range(len(lpth_ref),3):
                axs[1,j].axis('off')

        plt.show()


if __name__ == '__main__':
    eval_click()
