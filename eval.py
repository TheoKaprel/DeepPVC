import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np
import click
import random

from DeepPVC import helpers_data, helpers, Models,dataset, helpers_functions

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth', multiple = True)
@click.option('--input', '-i')
@click.option('-n',type=int, help = 'If no input is specified, choose the number of random images on which you want to test', default = 1)
@click.option('--dataset','dataset_path', help = 'path to the dataset folder in which to randomly select n images')
@click.option('--type', default = 'mhd', help = "mhd or mha", show_default = True)
@click.option('--merged', is_flag = True, default = False)
@click.option('--ref/--no-ref')
@click.option('--error', is_flag=True, help="Compute the MSE on the provided dataset")
@click.option('--plot', is_flag=True)
@click.option('-v', '--verbose', count=True)
def eval_click(pth, input, n, dataset_path,type,merged, ref, error, plot, verbose):

    if error:
        eval_error(pth, input, dataset_path,type, ref, verbose)
    if plot:
        eval_plot(pth, input, n, dataset_path,type,merged, ref, verbose)


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
    device = helpers.get_auto_device("cuda")

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
            test_dataset = torch.tensor(helpers_data.load_image(filename=input,is_ref=ref,type = type, nb_channels=params['input_channels'],noisy=params['with_noise']),device=device)
        elif dataset_path:
            test_dataset = dataset.CustomPVEProjectionsDataset(params=params, paths=[dataset_path])
        else:
            print('ERROR : no input nor dataset specified. You need to specify EITHER a --input /path/to/input OR a number -n 10 of image to select randomly in the dataset')
            exit(0)

        test_dataloader = DataLoader(dataset=test_dataset,batch_size=16,shuffle=False)
        with torch.no_grad():
            (MNRMSE,std_NRMSE), (MNMAE,std_NMAE) = helpers_functions.validation_errors(test_dataloader=test_dataloader,model=model,do_NMAE=True, do_NRMSE=True)
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



def eval_plot(lpth, input, n, dataset_path, type,merged, ref, verbose):
    device = helpers.get_auto_device("cuda")

    random_data_index = []

    dict_data = {'validation_losses': {}}
    lpth_ref = []

    for (pth_id,pth) in enumerate(lpth):
        pth_file = torch.load(pth, map_location=device)

        params = pth_file['params']
        data_normalisation = params['data_normalisation']
        pth_ref = params['ref']
        lpth_ref.append(pth_ref)

        model = Models.ModelInstance(params=params, from_pth=pth,resume_training=False)
        model.switch_device(device)
        model.switch_eval()

        if verbose>0:
            model.show_infos()
            if verbose > 1:
                model.plot_losses(save=False, wait=True, title=pth)
                dict_data['validation_losses'][pth_ref] = model.test_error


        if input:
            test_dataset = torch.tensor(helpers_data.load_image(filename=input,is_ref=ref,type=type, params=params),device=device).float()
        elif dataset_path:
            params['store_dataset']=True
            test_dataset = dataset.CustomPVEProjectionsDataset(params=params, paths=[dataset_path],dataset_type='test',filetype=type,merged=merged)
        else:
            print('ERROR : no input nor dataset specified. You need to specify EITHER a --input /path/to/input OR a number -n 10 of image to select randomly in the dataset')
            exit(0)

        # print(test_dataset.numpy_cpu_dataset.dtype)

        test_dataloader = DataLoader(dataset=test_dataset,batch_size=params['test_batchsize'],shuffle=False)

        if pth_id==0:
            N_data = len(test_dataloader.dataset)
            random_data_index = [random.randint(0, N_data - 1) for _ in range(n)]
            for id in random_data_index:
                dict_data[id] = {}
                dict_data[id]['PVE_noisy'] = test_dataloader.dataset[id][0,0,:,:].cpu().numpy()
                if ref:
                    dict_data[id]['PVE'] = test_dataloader.dataset[id][1, 0, :, :].cpu().numpy()
                    dict_data[id]['noPVE'] = test_dataloader.dataset[id][2, 0, :, :].cpu().numpy()


        for index in random_data_index:
            input_i = test_dataloader.dataset[index][0, :, :, :][None,None, :, :, :]
            with torch.no_grad():
                norm_input_i = helpers_data.compute_norm_eval(dataset_or_img=input_i, data_normalisation=data_normalisation)
                normed_input_i = helpers_data.normalize_eval(dataset_or_img=input_i, data_normalisation=data_normalisation,
                                                           norm=norm_input_i, params=model.params, to_torch=False)



                normed_output_i = model.forward(normed_input_i)
                denormed_output_i = helpers_data.denormalize_eval(dataset_or_img=normed_output_i,data_normalisation=data_normalisation,
                                                                norm=norm_input_i,params=model.params,to_numpy=False)


                dict_data[index][pth_ref] = [denormed_output_i[0, 0, :, :].float().cpu().numpy()]

                if verbose>2:
                    denoisedPVE = model.denoisedPVE
                    dict_data[index][pth_ref].append(denoisedPVE[0,0,:,:].cpu().numpy())


    if verbose>1:
        fig_test,ax_test = plt.subplots()
        for i ,(pth,test) in  enumerate(dict_data['validation_losses'].items()):
            ax_test.plot([e[0] for e in test],[e[1] for e in test],label = pth, linewidth= 2)
        ax_test.set_title('validation losses')
        plt.legend()

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
        if ref:
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
