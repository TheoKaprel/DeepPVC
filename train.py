import matplotlib.pyplot as plt
import torch
import time
import gc
import json as js
import os
import numpy as np
import click
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

import tracemalloc


from DeepPVC import dataset, Model_instance
from DeepPVC import helpers, helpers_params, helpers_functions,helpers_data

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
@click.option('--tensorboard','with_tensorboard', is_flag=True, default = False)
def train_onclick(json, resume_pth, user_param_str,user_param_float,user_param_int,user_param_bool,user_param_list,plot_at_end, output, output_folder,debug, with_tensorboard):

    train(json, resume_pth, user_param_str,user_param_float,user_param_int,user_param_bool,user_param_list,plot_at_end, output, output_folder, debug,with_tensorboard)


def train(json, resume_pth, user_param_str,user_param_float,user_param_int,user_param_bool,user_param_list,plot_at_end, output, output_folder, debug,with_tensorboard):
    t0 = time.time()
    params, start_epoch, ref = get_init_check_params(resume_pth=resume_pth,json=json,output=output)
    verbose=params['verbose']

    print(f"Initial mem ", torch.cuda.memory_allocated("cuda:0"))
    tracemalloc.start()

    # Update parameters specified in command line
    user_param_list = helpers_params.format_list_option(user_params=user_param_list)
    for user_param_to_modify in (user_param_str,user_param_float,user_param_int,user_param_bool,user_param_list):
        helpers_params.update_params_user_option(params,user_params=user_param_to_modify,is_resume=resume_pth)

    network_architecture = params['network']

    # torch.backends.cudnn.benchmark=True
    
    output_filename = f"{network_architecture}_{ref}_{start_epoch}_{start_epoch+params['n_epochs']}.pth"
    helpers_params.update_params_user_option(params, user_params=(("ref", ref),("output_folder", output_folder),("output_pth", output_filename)), is_resume=resume_pth)

    helpers_params.check_params(params)

    save_every_n_epoch,show_every_n_epoch,test_every_n_epoch = params['save_every_n_epoch'],params['show_every_n_epoch'],params['test_every_n_epoch']

    train_dataloader, test_dataloader, validation_dataloader, params = dataset.load_data(params)

    DeepPVEModel = Model_instance.ModelInstance(params=params, from_pth=resume_pth, resume_training=(resume_pth is not None))

    if resume_pth is not None:
        new_lr = None
        for up in user_param_float:
            if up[0]=='learning_rate':
                new_lr = up[1]
        DeepPVEModel.load_model(pth_path=resume_pth,new_lr=new_lr)

    if rank == 0:
        DeepPVEModel.show_infos()

    device = DeepPVEModel.device

    min_val_mae=np.infty

    DeepPVEModel.params['training_start_time'] = time.asctime()

    print(f"After model creation ", torch.cuda.memory_allocated(device))

    data_normalisation = params['data_normalisation']
    if with_tensorboard and (rank==0):
        writer=SummaryWriter(log_dir=os.path.join(output_folder,'runs/'+ref),flush_secs=60,filename_suffix=time.strftime("%Y_%m_%d_%Hh_%M_%S"))


    verbose_main_process=((params['jean_zay'] and idr_torch.rank == 0) or (not params['jean_zay'])) and (verbose>0)
    if verbose_main_process:
        print('Begining of training .....')

    # if debug:
    #     torch.autograd.set_detect_anomaly(True)

    for epoch in range(1,DeepPVEModel.n_epochs+1):
        if verbose_main_process:
            print(f'Epoch {DeepPVEModel.current_epoch}/{DeepPVEModel.n_epochs+DeepPVEModel.start_epoch- 1}')
        if debug:
            t_loading,timer_loading1=0,time.time()
            t_preopt,t_opt,t_test=0,0,0

        t0_epoch = time.time()
        # Optimisation loop
        DeepPVEModel.switch_train()
        # print(f"Epoch beggining ", torch.cuda.memory_allocated(device))

        for step,(batch_inputs,batch_targets) in enumerate(train_dataloader):
            if debug:
                timer_loading2 = time.time()

                t_step_begin = time.time()
                print("(begin) step {}   /   gpu {} (loading : {})".format(step,rank,timer_loading2-timer_loading1 ))

                t_loading+=timer_loading2-timer_loading1
                timer_preopt1=time.time()
            # print(f"Step beggining ", torch.cuda.memory_allocated(device))

            for key_inputs in batch_inputs.keys():
                batch_inputs[key_inputs] = batch_inputs[key_inputs].to(device, non_blocking=True)
            for key_targets in batch_targets.keys():
                batch_targets[key_targets] = batch_targets[key_targets].to(device, non_blocking=True)

            # for key_inputs in batch_inputs.keys():
            #     batch_inputs[key_inputs] = train_dataloader.dataset.pad(batch_inputs[key_inputs])
            # for key_targets in batch_targets.keys():
            #     batch_targets[key_targets] = train_dataloader.dataset.pad(batch_targets[key_targets])

            norm = helpers_data.compute_norm_eval(dataset_or_img=batch_inputs,data_normalisation=data_normalisation)
            batch_inputs = helpers_data.normalize_eval(dataset_or_img=batch_inputs,data_normalisation=data_normalisation,norm=norm,params=params,to_torch=False)
            batch_targets = helpers_data.normalize_eval(dataset_or_img=batch_targets,data_normalisation=data_normalisation,norm=norm,params=params,to_torch=False)

            # print(f"After batch->gpu ", torch.cuda.memory_allocated(device))

            if debug:
                t_preopt+=time.time()-timer_preopt1
                timer_opt1=time.time()

                if (step==0):
                    print(f'(gpu {rank}) batch_inputs shape : {[(k, v.shape) for (k,v) in batch_inputs.items()]}')
                    print(f'(gpu {rank}) batch_tagets shape : {[(k, v.shape) for (k,v) in batch_targets.items()]}')
                    print(f"(gpu {rank}) batch type : {batch_inputs[list(batch_inputs.keys())[0]].dtype}")
                    print(f' batch_inputs size (GiB) : {sum([b.element_size()*b.nelement()* 7.4506e-9 for b in batch_inputs.values()])}')
                    print(f' batch_targets size (GiB) : {sum([b.element_size()*b.nelement()* 7.4506e-9 for b in batch_targets.values()])}')
                    # print(f'norm : {norm[0]}')
                    # print(f' unet_denoiser size (GiB) : {sum([b.element_size()*b.nelement()* 7.4506e-9 for b in DeepPVEModel.UNet_denoiser.parameters()])}')

                    with torch.no_grad():
                        debug_output = DeepPVEModel.forward(batch=batch_inputs)
                        print(f'(gpu {rank}) output shape : {debug_output.shape}')
                        print(f'(gpu {rank}) output dtype : {debug_output.dtype}')
                        if (params['jean_zay']==False):
                            fig,ax = plt.subplots(max(len(batch_inputs.keys()),len(batch_targets.keys())),3)

                            i,j=np.random.randint(batch_inputs[list(batch_inputs.keys())[0]].shape[0]),50
                            for kk,key in enumerate(batch_inputs.keys()):
                                ax[kk,0].imshow(batch_inputs[key][i,j,:,:].float().detach().cpu().numpy())
                                ax[kk,0].set_title(key)

                            for kk, key in enumerate(batch_targets.keys()):
                                ax[kk,1].imshow(batch_targets[key][i,j,:,:].float().detach().cpu().numpy())
                                ax[kk,1].set_title(key)


                            ax[0,2].imshow(debug_output[i,j,:,:].float().detach().cpu().numpy())
                            ax[0,2].set_title('output')
                            plt.show()

            DeepPVEModel.input_data(batch_inputs=batch_inputs, batch_targets=batch_targets)
            # print(f"After input data ", torch.cuda.memory_allocated(device))

            del batch_inputs
            del batch_targets


            DeepPVEModel.optimize_parameters()



            # print(f"After empty cache ", torch.cuda.memory_allocated(device))

            # for obj in gc.get_objects():
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             print(type(obj), obj.size())
            #     except:
            #         pass

            if debug:
                t_opt += time.time() - timer_opt1

                timer_loading1 = time.time()
                t_step_end = time.time()
                print("(end) step {}   /   gpu {} ({})".format(step,rank, t_step_end-t_step_begin))


            if ((params['jean_zay']) and (time.time() - t0 >= 0.97*TIME_LIMIT_s)): # sauvegarde d'urgence
                if (idr_torch.rank == 0):
                    print('TIME LIMIT is close ! /!\ EMERGENCY SAVING /!\ ')
                    DeepPVEModel.params['training_duration'] = round(time.time() - t0)
                    emergency_output_filename = os.path.join(DeepPVEModel.output_folder, DeepPVEModel.output_pth.replace(".pth", f"_{DeepPVEModel.current_epoch}_emergency_saving.pth"))
                    DeepPVEModel.save_model(output_path=emergency_output_filename)
                exit(0)

            torch.cuda.empty_cache()

        if (DeepPVEModel.current_epoch % test_every_n_epoch == 0):
            if debug:
                timer_test=time.time()

            DeepPVEModel.switch_eval()

            MSE, MAE = helpers_functions.validation_errors(test_dataloader, DeepPVEModel, do_NRMSE=(params['validation_norm']=="L2"),
                                                                do_NMAE=(params['validation_norm']=="L1"))
            if params['validation_norm']=="L1":
                DeepPVEModel.test_error.append([DeepPVEModel.current_epoch, MAE.item()])
            elif params['validation_norm']=="L2":
                DeepPVEModel.test_error.append([DeepPVEModel.current_epoch, torch.sqrt(MSE).item()])

            if (validation_dataloader is not None):
                MSE_val,MAE_val = helpers_functions.validation_errors(validation_dataloader,DeepPVEModel,do_NRMSE=True, do_NMAE=True)
                RMSE_val,MAE_val = torch.sqrt(MSE_val).item(),MAE_val.item()
                DeepPVEModel.val_error_MSE.append([DeepPVEModel.current_epoch, RMSE_val])
                DeepPVEModel.val_error_MAE.append([DeepPVEModel.current_epoch, MAE_val])

                if rank==0:
                    print(f'valRMSE : {RMSE_val},  val MAE : {MAE_val}')

                if (MAE_val<min_val_mae and params['early_stopping'] and rank==0):
                    min_val_mae=MAE_val
                    DeepPVEModel.params['training_duration'] = round(time.time() - t0)
                    emergency_output_filename = os.path.join(DeepPVEModel.output_folder, DeepPVEModel.output_pth.replace(".pth",f"_early_stopping.pth"))
                    DeepPVEModel.save_model(output_path=emergency_output_filename)

            if verbose_main_process:
                print(f'Current mean test error =  {DeepPVEModel.test_error[-1][1]}')
            if debug:
                t_test=time.time() - timer_test

        if (DeepPVEModel.current_epoch % save_every_n_epoch==0 and DeepPVEModel.current_epoch!=DeepPVEModel.n_epochs and rank==0):
            current_time = round(time.time() - t0)
            DeepPVEModel.params['training_duration'] = current_time
            temp_output_filename = os.path.join(DeepPVEModel.output_folder,DeepPVEModel.output_pth[:-4]+f'_{DeepPVEModel.current_epoch}'+'.pth')
            DeepPVEModel.save_model(output_path=temp_output_filename)



        DeepPVEModel.update_epoch()


        if with_tensorboard and (rank==0):
            writer.add_scalar("Loss/G_train", DeepPVEModel.generator_losses[-1], epoch)
            writer.add_scalar("Loss/D_train", DeepPVEModel.discriminator_losses[-1], epoch)
            writer.add_scalar("Loss/test",DeepPVEModel.test_error[-1][1],DeepPVEModel.test_error[-1][0])
            tb_batch=next(iter(test_dataloader))[0:1,:,:,:,:].to(device)
            with torch.no_grad():
                norm = helpers_data.compute_norm_eval(dataset_or_img=tb_batch, data_normalisation=data_normalisation)
                tb_batch_n = helpers_data.normalize_eval(dataset_or_img=tb_batch, data_normalisation=data_normalisation,norm=norm, params=params, to_torch=False)
                out_tb_batch_n=DeepPVEModel.forward(batch=tb_batch_n)
                out_tb_batch=helpers_data.denormalize_eval(dataset_or_img=out_tb_batch_n,data_normalisation=data_normalisation,norm=norm,params=params,to_numpy=False)
                grid=tb_batch[0,:,0:1,:,:]
                grid=torch.concat((grid,out_tb_batch),dim=0)
                grid = grid / grid.max()
                writer.add_images('images',grid, epoch)

        if verbose_main_process:
            tf_epoch = time.time()
            print(f'time taken : {round(tf_epoch - t0_epoch,1)} s')

            if debug:
                print(f'loading time : {t_loading}')
                print(f'preopt time : {t_preopt}')
                print(f'opt time : {t_opt}')
                print(f'test time : {t_test}')
        # plt.show()

    if ((params['jean_zay'] and idr_torch.rank == 0) or (not params['jean_zay'])):
        tf = time.time()
        total_time = round(tf-t0)
        if verbose>0:
            print(f'Total training time : {total_time} s ({round(total_time/60/60,3)} h)')
        DeepPVEModel.params['training_endtime'] = time.asctime()
        DeepPVEModel.params['training_duration'] = total_time
        DeepPVEModel.save_model(save_json=True)

    if plot_at_end:
        DeepPVEModel.plot_losses(save = False, wait = False, title = params['ref'])

    if with_tensorboard and (rank==0):
        writer.flush()
        writer.close()


def get_init_check_params(resume_pth, json, output):
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

    return params,start_epoch,ref



if __name__ == '__main__':
    host = os.uname()[1]
    if (host !='siullus'):
        import idr_torch
        # get distributed configuration from Slurm environment
        NODE_ID = os.environ['SLURM_NODEID']
        MASTER_ADDR = os.environ['MASTER_ADDR'] if ("MASTER_ADDR" in os.environ) else os.environ['HOSTNAME']
        TIME_LIMIT = os.environ['SBATCH_TIMELIMIT']

        print(f"TIME_LIMIT is : {TIME_LIMIT}")
        TIME_LIMIT_s = 0
        TIME_LIMIT_split =  TIME_LIMIT.split(":")
        if len(TIME_LIMIT_split)==3:
            TIME_LIMIT_s = int(TIME_LIMIT_split[0])*60*60 + int(TIME_LIMIT_split[1])*60 + int(TIME_LIMIT_split[2])
        elif len(TIME_LIMIT_split)==2:
            TIME_LIMIT_s = int(TIME_LIMIT_split[0])*60 + int(TIME_LIMIT_split[1])
        print(f"i.e. {TIME_LIMIT_s} seconds")


        # display info
        if idr_torch.rank == 0:
            print(">>> Training on ", len(idr_torch.hostnames), " nodes and ", idr_torch.size,
                  " processes, master node is ", MASTER_ADDR)
        print("- Process {} corresponds to GPU {} of node {}".format(idr_torch.rank, idr_torch.local_rank, NODE_ID))

        dist.init_process_group(backend='nccl', init_method='env://', world_size=idr_torch.size, rank=idr_torch.rank)

        rank=idr_torch.rank
    else:
        rank=0

    train_onclick()


