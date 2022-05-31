import torch
import time
import json as js
import os
import numpy as np
import click
from data.dataset import load_data
from models.Pix2PixModel import PVEPix2PixModel
from utils import helpers,helpers_data,helpers_params,plots



CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)

@click.option('--json', help = 'JSON parameter file to start trainging FROM SCRATCH')
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
@click.option('--output', '-o', help='Output filename', default = None)
@click.option('--output_folder', '-f', help='Output folder ', default='.')
def train_onclick(json, resume, user_param_str,user_param_float,user_param_int,output, output_folder):
    train(json, resume, user_param_str,user_param_float,user_param_int,output, output_folder)


def train(json, resume, user_param_str,user_param_float,user_param_int,output, output_folder):
    if (json==None) and (resume ==None):
        print('ERROR : no json parameter file nor pth file to start/resume training')
        exit(0)

    if json and resume:
        print('WARNING : the json file will be ignored. The parameter file used will be the one from the pth file')



    if resume:
        is_resume = True
        device = helpers.get_auto_device("auto")
        checkpoint = torch.load(resume, map_location=device)
        params = checkpoint['params']
        params['start_pth'].append(resume)
        start_epoch = checkpoint['epoch']
    elif json:
        is_resume = False
        params_file = open(json).read()
        params = js.loads(params_file)
        params['start_pth'] = []
        start_epoch = 0
    else:
        print('ERROR : Absence of params not detected earlier ...')
        params = None
        is_resume=None
        start_epoch = 0
        exit(0)


    # Update parameters specified in command line
    helpers_params.update_params_user_option(params, user_params=user_param_str, is_resume=is_resume)
    helpers_params.update_params_user_option(params, user_params=user_param_float, is_resume=is_resume)
    helpers_params.update_params_user_option(params, user_params=user_param_int, is_resume=is_resume)


    if output:
        output_filename = f"pix2pix_{output}_{start_epoch}_{start_epoch+params['n_epochs']}.pth"
    else:
        output_filename = f"pix2pix_{start_epoch}_{start_epoch+params['n_epochs']}.pth"
    output_path = os.path.join(output_folder, output_filename)
    helpers_params.update_params_user_option(params, user_params=(("output_path", output_path),), is_resume=is_resume)


    helpers_params.check_params(params)

    save_every_n_epoch = params['save_every_n_epoch']
    show_every_n_epoch = params['show_every_n_epoch']
    test_every_n_epoch = params['test_every_n_epoch']

    device = torch.device(params['device'])

    train_dataloader, test_dataloader = load_data(dataset_path=params['dataset_path'],
                                                  training_batchsize=params['training_batchsize'],
                                                  testing_batchsize=params['test_batchsize'],
                                                  prct_train=params['training_prct'],
                                                  normalisation = params['data_normalisation'],
                                                  device = device)
    nb_train_data = len(train_dataloader.dataset)
    testdataset = test_dataloader.dataset
    nb_test_data = len(testdataset)

    print(f'Number of training data : {nb_train_data}')
    print(f'Number of testing data : {nb_test_data}')

    DeepPVEModel = PVEPix2PixModel(params, is_resume)
    DeepPVEModel.params['nb_training_data'] = nb_train_data
    DeepPVEModel.params['nb_testing_data'] = nb_test_data

    DeepPVEModel.show_infos()


    DeepPVEModel.switch_train()

    DeepPVEModel.params['training_start_time'] = time.asctime()

    t0 = time.time()
    for epoch in range(DeepPVEModel.n_epochs):
        print(f'Epoch {DeepPVEModel.current_epoch}/{DeepPVEModel.n_epochs+DeepPVEModel.start_epoch}')

        # Optimisation loop
        DeepPVEModel.switch_train()
        for step,batch in enumerate(train_dataloader):
            DeepPVEModel.input_data(batch)
            DeepPVEModel.optimize_parameters()


        DeepPVEModel.update_epoch()

        if (DeepPVEModel.current_epoch % test_every_n_epoch == 0):
            DeepPVEModel.switch_eval()
            MSE= 0
            for test_it,batch in enumerate(test_dataloader):
                DeepPVEModel.input_data(batch)
                fakePVE = DeepPVEModel.test(DeepPVEModel.truePVE)
                MSE += torch.mean((DeepPVEModel.truePVfree - fakePVE)**2).item()
            DeepPVEModel.test_mse.append([DeepPVEModel.current_epoch, MSE])
            print(f'Current MSE  =  {MSE}')


        if (DeepPVEModel.current_epoch % show_every_n_epoch==0):
            DeepPVEModel.plot_losses()
            id_test = np.random.randint(0,nb_test_data)
            testdata = testdataset[id_test]
            input = testdata[0,:,:]
            input = input[None, None, :,:]
            output = DeepPVEModel.test(input)
            imgs = torch.cat((testdata[None, :,:,:], output), dim=1)
            plots.show_images_profiles(imgs, profile=True)


        if (DeepPVEModel.current_epoch % save_every_n_epoch==0):
            DeepPVEModel.save_model()


    tf = time.time()
    total_time = round(tf-t0)
    print(f'Total training time : {total_time} s')
    DeepPVEModel.params['training_endtime'] = total_time



    DeepPVEModel.save_model()
    DeepPVEModel.plot_losses()



if __name__ == '__main__':
    train_onclick()


