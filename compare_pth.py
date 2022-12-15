import torch
import matplotlib.pyplot as plt
import numpy as np
import click
import glob
import random


from DeepPVC import helpers_data, helpers, Models, helpers_params
from DeepPVC import dataset as ds

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth', multiple = True) # 'path/to/saved/model.pth'
@click.option('--proj', is_flag = True, default = False)
@click.option('--input', '-i', multiple = True)
@click.option('-n', help = 'If no input is specified, choose the number of random images on which you want to test')
@click.option('--dataset',multiple = True, help = 'path to the dataset folder in which to randomly select n images')
@click.option('--ref/--no-ref', default = True)
@click.option('--type', default = 'mhd', help = "mhd or mha", show_default = True)
@click.option('--losses', is_flag = True, default = False)
@click.option('--calc_mse', is_flag = True, default = False, help = 'Compute mse on --dataset')
def compare_proj_pth_click(pth,proj, input, n, dataset, ref,type, losses, calc_mse):
    compare_proj_pth(pth,proj, input, n, dataset, ref,type, losses, calc_mse)



def compare_proj_pth(pth,proj, input, n, dataset, ref,type, losses, calc_mse):

    # load models
    device = helpers.get_auto_device("cpu")
    nPth = len(pth)
    if nPth<2:
        print('ERROR : the number of pth file should be > 1 since this code is meant to compare 2 or more pth on projections')
        exit(0)

    list_models = []
    list_params = []
    list_refs = []
    for idpth in range(nPth):
        one_pth = pth[idpth]
        pth_file = torch.load(one_pth, map_location=device)
        params = pth_file['params']
        list_params.append(params)
        list_refs.append(params['ref'])

        model = Models.ModelInstance(params=params, from_pth=pth_file,resume_training=False)
        model.switch_device("cpu")
        model.switch_eval()
        list_models.append(model)
        if losses:
            model.plot_losses(save=False, wait = True, title = one_pth)


    if losses:
        plt.show()

    if proj:
        # load projections
        if input:
            list_of_images = list(input)
        elif (n and dataset):
            n = int(n)
            list_of_all_images = []
            for d in dataset:
                list_of_all_images = [*list_of_all_images, *glob.glob(f'{d}/?????_PVE.mhd')]
            Nimages = len(list_of_all_images)
            list_of_all_images = [list_of_all_images[i][:-8] for i in range(Nimages)]
            list_of_images = random.sample(list_of_all_images, n)
        else:
            print(
                'ERROR : no input nor n/dataset specified. You need to specify EITHER a --input /path/to/input OR a number -n 10 of image and a --dataset /pth/to/dataset to select randomly in the dataset')
            list_of_images = []
            exit(0)



    if calc_mse:
        list_dataloaders = []
        for one_dataset in dataset:
            test_dataset = ds.construct_dataset_from_path(dataset_path=one_dataset, datatype=type)
            nb_testing_data = test_dataset.shape[0]
            print(f'nb_test : {nb_testing_data}')


            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=list_models[0].params['test_batchsize'], shuffle=True)
            list_dataloaders.append([test_dataloader,nb_testing_data, one_dataset])


        for m in range(nPth):
            model = list_models[m]
            print(model.test_mse)

            model.params['MSE'] = []
            for test_dataloader,nb_testing_data,dataset_filename in list_dataloaders:

                mse = 0
                with torch.no_grad():
                    for test_it, batch in enumerate(test_dataloader):
                        normalized_batch = helpers_data.normalize(batch,normtype=model.params['data_normalisation'], norm=model.params['norm'], to_torch=False, device=device)
                        fakePVfree = model.forward(batch=normalized_batch)

                        denormalized_input = helpers_data.denormalize(model.truePVfree,normtype=model.params['data_normalisation'], norm=model.params['norm'],to_numpy=True)
                        denormalized_output = helpers_data.denormalize(fakePVfree, normtype=model.params['data_normalisation'],norm=model.params['norm'], to_numpy=True)


                        mse+= np.sum(np.sum((denormalized_output - denormalized_input) ** 2, axis=(1, 2, 3)) / np.sum(denormalized_input **2, axis=(1,2,3))) / nb_testing_data
                model.params['MSE'].append([dataset_filename,mse])

            model.save_model(pth[m], save_json=False)


    helpers_params.make_and_print_params_info_table([model.params for model in list_models])


    if proj:
        for img in list_of_images:
            is_ref = ref
            input_array = helpers_data.load_image(img, is_ref, type)

            projs_DeepPVC = np.zeros((n,128,128))

            for idpth in range(nPth):
                model = list_models[idpth]
                normalisation = model.params['data_normalisation']
                norm = model.params['norm']

                with torch.no_grad():
                    normalized_input_tensor = helpers_data.normalize(dataset_or_img=input_array, normtype=normalisation,norm=norm,to_torch=True, device='cpu')
                    output_tensor = model.forward(normalized_input_tensor)

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

            lmse = []
            for k in range(nPth):
                ax[1,k].imshow(projs_DeepPVC[k,:,:], vmin=vmin, vmax=vmax)
                # ax[1,k].imshow((projs_DeepPVC[k,:,:] - input_array_sq[1,:,:])**2)
                ax[1,k].set_title(pth[k])
                mse = np.sum((projs_DeepPVC[k,:,:] - input_array_sq[1,:,:])**2) / np.sum(input_array_sq[1,:,:]**2)
                ax[1,k].set_ylabel("MSE = {}".format(mse))
                lmse.append(mse)
            plt.suptitle(img)

            fig,ax = plt.subplots()
            ax.bar(list_refs, lmse)
            ax.set_ylabel('MSE')

            plt.show()






if __name__ == '__main__':
    compare_proj_pth_click()
