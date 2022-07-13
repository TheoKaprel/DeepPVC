import torch
import time
import json as js
import os
import numpy as np
import click

from DeepPVC import dataset, Pix2PixModel, helpers, helpers_data, helpers_params, plots


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)

@click.option('--json', help = 'JSON parameter file to start training FROM SCRATCH')
@click.option('--resume', help = 'PTH file from which to RESUME training')
@click.option('--user_param_str', '-ps',
              help='overwrite str parameter of the json file',
              multiple=True, type=(str, str))
@click.option('--user_param_float', '-pf',
              help='overwrite numeric parameter of the json file',
              multiple=True, type=(str, float))
@click.option('--user_param_int', '-pi',
              help='overwrite numeric int parameter of the json file',
              multiple=True, type=(str, int))
@click.option('--plot_at_end', is_flag = True, default = False)
@click.option('--output', '-o', help='Output Reference. Highly recommended to specify one.', default = None)
@click.option('--output_folder', '-f', help='Output folder ', default='.')
def train_onclick(json, resume, user_param_str,user_param_float,user_param_int,plot_at_end, output, output_folder):
    train(json, resume, user_param_str,user_param_float,user_param_int,plot_at_end, output, output_folder)


def train(json, resume, user_param_str,user_param_float,user_param_int,plot_at_end, output, output_folder):
    if (json==None) and (resume ==None):
        print('ERROR : no json parameter file nor pth file to start/resume training')
        exit(0)

    if json and resume:
        print('WARNING : the json file will be ignored. The parameter file used will be the one contained in the pth file')

    if resume:
        is_resume = True
        device = helpers.get_auto_device("auto")
        checkpoint = torch.load(resume, map_location=device)
        params = checkpoint['params']
        params['start_pth'].append(resume)
        start_epoch = checkpoint['epoch']
        ref = params['ref']
    elif json:
        is_resume = False
        params_file = open(json).read()
        params = js.loads(params_file)
        params['start_pth'] = []
        start_epoch = 0

        if output:
            ref = output
        else:
            ref = 'NOREF'
    else:
        print('ERROR : Absence of params not detected earlier my bad ...')
        params = None
        is_resume=None
        start_epoch = 0
        exit(0)


    # Update parameters specified in command line
    helpers_params.update_params_user_option(params, user_params=user_param_str, is_resume=is_resume)
    helpers_params.update_params_user_option(params, user_params=user_param_float, is_resume=is_resume)
    helpers_params.update_params_user_option(params, user_params=user_param_int, is_resume=is_resume)




    output_filename = f"pix2pix_{ref}_{start_epoch}_{start_epoch+params['n_epochs']}.pth"
    helpers_params.update_params_user_option(params, user_params=(("ref", ref),("output_folder", output_folder),("output_pth", output_filename)), is_resume=is_resume)

    helpers_params.check_params(params)

    save_every_n_epoch,show_every_n_epoch,test_every_n_epoch = params['save_every_n_epoch'],params['show_every_n_epoch'],params['test_every_n_epoch']

    train_dataloader, test_dataloader, params = dataset.load_data(params)



    DeepPVEModel = Pix2PixModel.PVEPix2PixModel(params, is_resume)
    DeepPVEModel.show_infos()

    DeepPVEModel.params['training_start_time'] = time.asctime()

    t0 = time.time()


    DeepPVEModel.switch_eval()
    with torch.no_grad():
        # the data which will be used for show/test
        nb_testing_data = params['nb_testing_data']
        testdataset = test_dataloader.dataset
        id_test = np.random.randint(0, nb_testing_data)
        show_test_data = testdataset[id_test][None,:,:,:] #(1,2,128,128)
        show_test_PVE = show_test_data[:,0, :, :][:, None, :, :] # (1,1,128,128)
        show_test_denormalized_PVE_PVfree = helpers_data.denormalize(show_test_data, normtype=params['data_normalisation'],norm=params['norm'], to_numpy=True)  # (1,2,128,128)
        show_test_keep_data = show_test_denormalized_PVE_PVfree
        show_test_keep_labels = ['PVE', 'PVfree']
        show_test_fakePVfree = DeepPVEModel.Generator(show_test_PVE)  # (1,1,128,128)
        denormalized_output = helpers_data.denormalize(show_test_fakePVfree, normtype=params['data_normalisation'], norm=params['norm'],to_numpy=True)  # (1,1,128,128)
        show_test_keep_data = np.concatenate((show_test_keep_data, denormalized_output), axis=1)  # (1,2+n,128,128)
        show_test_keep_labels.append('Pix2Pix:0')


    print('Begining of the training .....')
    for epoch in range(1,DeepPVEModel.n_epochs+1):
        print(f'Epoch {DeepPVEModel.current_epoch}/{DeepPVEModel.n_epochs+DeepPVEModel.start_epoch- 1}')

        # Optimisation loop
        DeepPVEModel.switch_train()
        for step,batch in enumerate(train_dataloader):
            DeepPVEModel.input_data(batch)
            DeepPVEModel.optimize_parameters()

        if (DeepPVEModel.current_epoch % test_every_n_epoch == 0):
            DeepPVEModel.switch_eval()
            MSE = 0
            with torch.no_grad():
                for test_it,batch in enumerate(test_dataloader):
                    DeepPVEModel.input_data(batch)
                    fakePVfree = DeepPVEModel.Generator(DeepPVEModel.truePVE)

                    denormalized_target = helpers_data.denormalize(DeepPVEModel.truePVfree, normtype=params['data_normalisation'],norm=params['norm'], to_numpy=True)
                    denormalized_output = helpers_data.denormalize(fakePVfree, normtype=params['data_normalisation'],norm=params['norm'], to_numpy=True)
                    MSE += np.sum(np.mean((denormalized_output - denormalized_target)**2, axis=(2,3)))/nb_testing_data

            DeepPVEModel.test_mse.append([DeepPVEModel.current_epoch, MSE])
            print(f'Current MSE  =  {MSE}')


        if (DeepPVEModel.current_epoch % show_every_n_epoch==0):
            DeepPVEModel.switch_eval()
            with torch.no_grad():
                show_test_output = DeepPVEModel.Generator(show_test_PVE) # (1,1,128,128)
                show_test_denormalized_output = helpers_data.denormalize(show_test_output, normtype=params['data_normalisation'],norm=params['norm'], to_numpy=True) # (1,1,128,128)
                show_test_keep_data = np.concatenate((show_test_keep_data,show_test_denormalized_output), axis=1) # (1,2+n,128,128)
                show_test_keep_labels.append(f'Pix2Pix:{DeepPVEModel.current_epoch}')


        if (DeepPVEModel.current_epoch % save_every_n_epoch==0 and DeepPVEModel.current_epoch!=DeepPVEModel.n_epochs):
            current_time = round(time.time() - t0)
            DeepPVEModel.params['training_duration'] = current_time
            temp_output_filename = os.path.join(DeepPVEModel.output_folder,DeepPVEModel.output_pth[:-4]+f'_{DeepPVEModel.current_epoch}'+'.pth')
            DeepPVEModel.save_model(output_path=temp_output_filename)

        DeepPVEModel.update_epoch()


    tf = time.time()
    total_time = round(tf-t0)
    print(f'Total training time : {total_time} s')
    DeepPVEModel.params['training_endtime'] = time.asctime()
    DeepPVEModel.params['training_duration'] = total_time

    if show_every_n_epoch<DeepPVEModel.n_epochs:
        plots.show_images_profiles(show_test_keep_data[0,:,:,:],profile = True, save=True,folder = DeepPVEModel.params['output_folder'], is_tensor=False, title = f'Saved Images {ref}', labels=show_test_keep_labels)

    DeepPVEModel.save_model(save_json=True)

    if plot_at_end:
        DeepPVEModel.plot_losses(save = False, wait = False, title = params['ref'])


if __name__ == '__main__':
    train_onclick()


