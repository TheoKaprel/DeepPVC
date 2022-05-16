from data.dataset import load_data
from models.Pix2PixModel import PVEPix2PixModel
from utils.helpers_params import *
import time
import json
import os
import click



CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('json_filename', type=click.Path(exists=True, file_okay=True, dir_okay=False))
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
def train_onclick(json_filename, user_param_str,user_param_float,user_param_int,output, output_folder):
    train(json_filename, user_param_str,user_param_float,user_param_int,output, output_folder)


def train(json_filename, user_param_str,user_param_float,user_param_int,output, output_folder):
    params_file = open(json_filename).read()
    params = json.loads(params_file)


    # Update parameters specified in command line
    update_params_user_option(params, user_params=user_param_str)
    update_params_user_option(params, user_params=user_param_float)
    update_params_user_option(params, user_params=user_param_int)


    if output:
        output_filename = f"pix2pix_{output}_{params['n_epochs']}.pth"
    else:
        output_filename = f"pix2pix_{params['n_epochs']}.pth"
    output_path = os.path.join(output_folder, output_filename)
    update_params_user_option(params, user_params=(("output_path", output_path),))

    check_params(params)



    train_dataloader, test_dataloader = load_data(dataset_path=params['dataset_path'],
                                                  training_batchsize=params['training_batchsize'],
                                                  testing_batchsize=params['test_batchsize'],
                                                  prct_train=params['training_prct'])



    DeepPVEModel = PVEPix2PixModel(params)

    DeepPVEModel.show_infos()



    DeepPVEModel.params['training_start_time'] = time.asctime()

    t0 = time.time()
    for epoch in range(DeepPVEModel.n_epochs):
        print(f'Epoch {DeepPVEModel.current_epoch}/{DeepPVEModel.n_epochs+DeepPVEModel.start_epoch}')
        for step,batch in enumerate(train_dataloader):
            print(f'step {step}/{len(train_dataloader)-1}.........................')

            DeepPVEModel.input_data(batch)

            DeepPVEModel.optimize_parameters()

            if (step % DeepPVEModel.display_step == 0) & step!=0:
                DeepPVEModel.display()

        DeepPVEModel.update_epoch()


    tf = time.time()
    total_time = round(tf-t0)
    print(f'Total training time : {total_time} s')
    DeepPVEModel.params['training_endtime'] = total_time

    DeepPVEModel.save_model()
    DeepPVEModel.plot_losses()


if __name__ == '__main__':
    train_onclick()


