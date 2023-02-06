import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast
import time
import json as js
import os
import click

from DeepPVC import dataset, Models
from DeepPVC import helpers, helpers_params, helpers_functions,helpers_data, helpers_data_parallelism

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
@click.option('--user_param_list', '-pl',
              help="""overwrite list parameter of the json file. Ex : -pl recon_loss_denoiser "['L1','Poisson']" """,
              multiple=True, type=(str, str))
@click.option('--plot_at_end', is_flag = True, default = False)
@click.option('--output', '-o', help='Output Reference. Highly recommended to specify one.', default = None)
@click.option('--output_folder', '-f', help='Output folder ', default='.')
@click.option('--debug', is_flag=True, default = False)
def train_onclick(json, resume_pth, user_param_str,user_param_float,user_param_int,user_param_bool,user_param_list,plot_at_end, output, output_folder,debug):

    train(json, resume_pth, user_param_str,user_param_float,user_param_int,user_param_bool,user_param_list,plot_at_end, output, output_folder, debug)


def train(json, resume_pth, user_param_str,user_param_float,user_param_int,user_param_bool,user_param_list,plot_at_end, output, output_folder, debug):
    if (json==None) and (resume_pth ==None):
        print('ERROR : no json parameter file nor pth file to start/resume training')
        exit(0)


    if resume_pth is not None:
        device = helpers.get_auto_device("auto")
        checkpoint = torch.load(resume_pth, map_location=device)
        if json:
            params_file = open(json).read()
            params = js.loads(params_file)
            params['start_pth'] = [resume_pth]
        else:
            params = checkpoint['params']
            params['start_pth'].append(resume_pth)

        if output:
            ref=output
        else:
            ref=checkpoint['params']['ref']

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
    user_param_list = helpers_params.format_list_option(user_params=user_param_list)
    for user_param_to_modify in (user_param_str,user_param_float,user_param_int,user_param_bool,user_param_list):
        helpers_params.update_params_user_option(params,user_params=user_param_to_modify,is_resume=resume_pth)

    network_architecture = params['network']

    output_filename = f"{network_architecture}_{ref}_{start_epoch}_{start_epoch+params['n_epochs']}.pth"
    helpers_params.update_params_user_option(params, user_params=(("ref", ref),("output_folder", output_folder),("output_pth", output_filename)), is_resume=resume_pth)

    helpers_params.check_params(params)

    save_every_n_epoch,show_every_n_epoch,test_every_n_epoch = params['save_every_n_epoch'],params['show_every_n_epoch'],params['test_every_n_epoch']

    train_normalized_dataloader, test_dataloader, params = dataset.load_data(params)

    DeepPVEModel = Models.ModelInstance(params=params, from_pth=resume_pth, resume_training=(resume_pth is not None))

    if params['jean_zay']:
        if resume_pth is not None:
            DeepPVEModel.load_model(pth_path=resume_pth)

    DeepPVEModel.show_infos()

    device = DeepPVEModel.device

    DeepPVEModel.params['training_start_time'] = time.asctime()

    t0 = time.time()

    data_normalisation = params['data_normalisation']

    print('Begining of training .....')
    for epoch in range(1,DeepPVEModel.n_epochs+1):
        if ((params['jean_zay'] and idr_torch.rank == 0) or (not params['jean_zay'])):
            print(f'Epoch {DeepPVEModel.current_epoch}/{DeepPVEModel.n_epochs+DeepPVEModel.start_epoch- 1}')
        t0_epoch=time.time()
        # Optimisation loop
        DeepPVEModel.switch_train()
        t_loading,timer_loading1=0,time.time()
        t_preopt,t_opt=0,0

        for step,batch in enumerate(train_normalized_dataloader):
            timer_loading2=time.time()
            t_loading+=timer_loading2-timer_loading1

            timer_preopt1=time.time()
            norm = helpers_data.compute_norm_eval(dataset_or_img=batch,data_normalisation=data_normalisation)
            batch = helpers_data.normalize_eval(dataset_or_img=batch,data_normalisation=data_normalisation,norm=norm,params=params,to_torch=False)

            batch = batch.to(device,non_blocking=True)
            DeepPVEModel.input_data(batch)

            t_preopt+=time.time()-timer_preopt1

            timer_opt1=time.time()
            DeepPVEModel.optimize_parameters()
            t_opt+=time.time() - timer_opt1

            if debug:
                if step==0:
                    random_sample = torch.randint(0,batch.shape[0],(1,)).item()
                    print(f'batch shape : {batch.shape}')
                    print(f'batch type : {batch.dtype}')
                    fig,ax = plt.subplots(batch.shape[1]+1,batch.shape[2],squeeze=False)
                    for i in range(batch.shape[1]):
                        for j in range(batch.shape[2]):
                            ax[i,j].imshow(batch[random_sample,i,j,:,:].float().detach().cpu().numpy())
                    with torch.no_grad():
                        with autocast():
                            debug_output = DeepPVEModel.forward(batch=batch)
                        print(f'output shape : {debug_output.shape}')
                    ax[batch.shape[1],0].imshow(debug_output[random_sample,0,:,:].float().detach().cpu().numpy())
                    plt.show()

            timer_loading1 = time.time()

        if (DeepPVEModel.current_epoch % test_every_n_epoch == 0):
            timer_test=time.time()
            DeepPVEModel.switch_eval()


            MNRMSE, MNMAE = helpers_functions.validation_errors(test_dataloader, DeepPVEModel, do_NRMSE=(params['validation_norm']=="L2"),
                                                                do_NMAE=(params['validation_norm']=="L1"))
            if params['validation_norm']=="L1":
                DeepPVEModel.test_error.append([DeepPVEModel.current_epoch, MNMAE.item()])
            elif params['validation_norm']=="L2":
                DeepPVEModel.test_error.append([DeepPVEModel.current_epoch, MNRMSE.item()])

            if ((params['jean_zay'] and idr_torch.rank == 0) or (not params['jean_zay'])):
                print(f'Current mean validation error =  {DeepPVEModel.test_error[-1][1]}')

            t_test=time.time() - timer_test

        if (DeepPVEModel.current_epoch % save_every_n_epoch==0 and DeepPVEModel.current_epoch!=DeepPVEModel.n_epochs):
            if ((params['jean_zay'] and idr_torch.rank == 0) or (not params['jean_zay'])):
                current_time = round(time.time() - t0)
                DeepPVEModel.params['training_duration'] = current_time
                temp_output_filename = os.path.join(DeepPVEModel.output_folder,DeepPVEModel.output_pth[:-4]+f'_{DeepPVEModel.current_epoch}'+'.pth')
                DeepPVEModel.save_model(output_path=temp_output_filename)

        DeepPVEModel.update_epoch()
        if ((params['jean_zay'] and idr_torch.rank == 0) or (not params['jean_zay'])):
            tf_epoch = time.time()
            print(f'time taken : {round(tf_epoch - t0_epoch,1)} s')

            print(f'loading time : {t_loading}')
            print(f'preopt time : {t_preopt}')
            print(f'opt time : {t_opt}')
            print(f'test time : {t_test}')


    if ((params['jean_zay'] and idr_torch.rank == 0) or (not params['jean_zay'])):
        tf = time.time()
        total_time = round(tf-t0)
        print(f'Total training time : {total_time} s ({round(total_time/60/60,3)} h)')
        DeepPVEModel.params['training_endtime'] = time.asctime()
        DeepPVEModel.params['training_duration'] = total_time
        DeepPVEModel.save_model(save_json=True)

    if plot_at_end:
        DeepPVEModel.plot_losses(save = False, wait = False, title = params['ref'])

if __name__ == '__main__':
    host = os.uname()[1]
    if (host !='siullus'):
        import idr_torch
        # get distributed configuration from Slurm environment
        NODE_ID = os.environ['SLURM_NODEID']
        MASTER_ADDR = os.environ['MASTER_ADDR']

        # display info
        if idr_torch.rank == 0:
            print(">>> Training on ", len(idr_torch.hostnames), " nodes and ", idr_torch.size,
                  " processes, master node is ", MASTER_ADDR)
        print("- Process {} corresponds to GPU {} of node {}".format(idr_torch.rank, idr_torch.local_rank, NODE_ID))

    train_onclick()


