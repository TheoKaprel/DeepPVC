import torch
import time
import json as js
import os
import click

from DeepPVC import dataset, helpers, helpers_params, helpers_functions, Models


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)

@click.option('--json', help = 'JSON parameter file to start training FROM SCRATCH')
@click.option('--resume', 'resume_pth', help = 'PTH file from which to RESUME training')
@click.option('--user_param_str', '-ps',
              help='overwrite str parameter of the json file',
              multiple=True, type=(str, str))
@click.option('--user_param_float', '-pf',
              help='overwrite numeric parameter of the json file',
              multiple=True, type=(str, float))
@click.option('--user_param_int', '-pi',
              help='overwrite numeric int parameter of the json file',
              multiple=True, type=(str, int))
@click.option('--user_param_bool', '-pb',
              help='overwrite boolean parameter of the json file',
              multiple=True, type=(str, bool))
@click.option('--plot_at_end', is_flag = True, default = False)
@click.option('--output', '-o', help='Output Reference. Highly recommended to specify one.', default = None)
@click.option('--output_folder', '-f', help='Output folder ', default='.')
def train_onclick(json, resume_pth, user_param_str,user_param_float,user_param_int,user_param_bool,plot_at_end, output, output_folder):
    train(json, resume_pth, user_param_str,user_param_float,user_param_int,user_param_bool,plot_at_end, output, output_folder)


def train(json, resume_pth, user_param_str,user_param_float,user_param_int,user_param_bool,plot_at_end, output, output_folder):
    if (json==None) and (resume_pth ==None):
        print('ERROR : no json parameter file nor pth file to start/resume training')
        exit(0)


    if resume_pth:
        device = helpers.get_auto_device("auto")
        checkpoint = torch.load(resume_pth, map_location=device)
        if json:
            params_file = open(json).read()
            params = js.loads(params_file)
            params['start_pth'] = [resume_pth]
            if output:
                ref = output
            else:
                ref = checkpoint['params']['ref']
        else:
            params = checkpoint['params']
            params['start_pth'].append(resume_pth)
            ref = params['ref']
        start_epoch = checkpoint['epoch']

    elif json and not(resume_pth):
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
        params,start_epoch,ref = None,0,None
        exit(0)


    # Update parameters specified in command line
    helpers_params.update_params_user_option(params, user_params=user_param_str, is_resume=resume_pth)
    helpers_params.update_params_user_option(params, user_params=user_param_float, is_resume=resume_pth)
    helpers_params.update_params_user_option(params, user_params=user_param_int, is_resume=resume_pth)
    helpers_params.update_params_user_option(params, user_params=user_param_bool, is_resume=resume_pth)


    network_architecture = params['network']

    output_filename = f"{network_architecture}_{ref}_{start_epoch}_{start_epoch+params['n_epochs']}.pth"
    helpers_params.update_params_user_option(params, user_params=(("ref", ref),("output_folder", output_folder),("output_pth", output_filename)), is_resume=resume_pth)

    helpers_params.check_params(params)

    save_every_n_epoch,show_every_n_epoch,test_every_n_epoch = params['save_every_n_epoch'],params['show_every_n_epoch'],params['test_every_n_epoch']

    train_dataloader, test_dataloader, params = dataset.load_data(params)


    DeepPVEModel = Models.ModelInstance(params=params, from_pth=resume_pth)
    DeepPVEModel.show_infos()

    DeepPVEModel.params['training_start_time'] = time.asctime()

    t0 = time.time()


    print('Begining of training .....')
    for epoch in range(1,DeepPVEModel.n_epochs+1):
        print(f'Epoch {DeepPVEModel.current_epoch}/{DeepPVEModel.n_epochs+DeepPVEModel.start_epoch- 1}')

        # Optimisation loop
        DeepPVEModel.switch_train()
        for step,batch in enumerate(train_dataloader):
            DeepPVEModel.input_data(batch)
            DeepPVEModel.optimize_parameters()

        if (DeepPVEModel.current_epoch % test_every_n_epoch == 0):
            DeepPVEModel.switch_eval()

            if params['validation_norm']=="L1":
                MNRMSE,MNMAE = helpers_functions.validation_errors(test_dataloader,DeepPVEModel,do_NRMSE=False, do_NMAE=True)
                DeepPVEModel.test_error.append([DeepPVEModel.current_epoch, MNMAE])
            if params['validation_norm']=="L2":
                MNRMSE,MNMAE = helpers_functions.validation_errors(test_dataloader,DeepPVEModel,do_NRMSE=True, do_NMAE=False)
                DeepPVEModel.test_error.append([DeepPVEModel.current_epoch, MNRMSE])

            print(f'Current mean validation error =  {DeepPVEModel.test_error[-1][1]}')


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

    DeepPVEModel.save_model(save_json=True)

    if plot_at_end:
        DeepPVEModel.plot_losses(save = False, wait = False, title = params['ref'])


if __name__ == '__main__':
    train_onclick()


