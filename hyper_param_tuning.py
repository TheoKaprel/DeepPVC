#!/usr/bin/env python3
import torch.distributed as dist
from DeepPVC import dataset,Model_instance,helpers_functions

import click
import optuna
import json
import os

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--n_trials", type=int, default=30)
@click.option("--parambase")
@click.option("--tune")
def optuna_study(n_trials, parambase, tune):
    params_file = open(parambase).read()
    params = json.loads(params_file)
    tune = tune.split(',')
    print(tune)
    objective = lambda single_trial: objective_w_params(single_trial= single_trial,params=params, tune=tune)

    if rank==0:
        print(params_file)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials)
    else:
        for _ in range(n_trials):
            try:
                objective(None)
            except optuna.TrialPruned:
                pass

    if rank==0:
        best_trial = study.best_trial
        print('*'*50)
        print('Final Result : ')
        for key, value in best_trial.params.items():
            print("{}: {}\n".format(key, value))


def objective_w_params(single_trial, params, tune):
    if params['jean_zay']:
        trial=optuna.integration.TorchDistributedTrial(single_trial)
    else:
        trial=single_trial

    for param_to_tune in tune:
        if param_to_tune=="with_rec_fp":
            params['with_rec_fp'] = trial.suggest_categorical('with_rec_fp', choices=[True,False])
        if param_to_tune == "denoise":
            params['denoise'] = trial.suggest_categorical('denoise', choices=[True, False])
        if param_to_tune == "data_normalisation":
            params['data_normalisation'] = trial.suggest_categorical('data_normalisation', choices=["none", "sino_sum", "3d_max"])
        if param_to_tune=="n_epochs":
            params['n_epochs'] = trial.suggest_int('n_epochs',10,200)
        if param_to_tune == 'learning_rate':
            params['learning_rate']=trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        if param_to_tune=="init_feature_kernel":
            params['init_feature_kernel'] = trial.suggest_discrete_uniform('init_feature_kernel', low=3,high=9,q=2)
        if param_to_tune == "final_feature_kernel":
            params['final_feature_kernel'] = trial.suggest_discrete_uniform('final_feature_kernel', low=1, high=9, q=2)
        if param_to_tune=="nb_ed_layers":
            params['nb_ed_layers']=trial.suggest_int('nb_ed_layers',1,1)
        if param_to_tune == "hidden_channels_unet":
            params['hidden_channels_unet'] = trial.suggest_categorical('hidden_channels_unet', choices=[4, 8, 16, 32, 64])
        if param_to_tune=='loss_denoiser':
            loss_denoiser=[0]
            loss_denoiser[0]=trial.suggest_categorical('loss_denoiser', choices=["L1", "L2"])
            params['loss_denoiser']=loss_denoiser
        if param_to_tune=='loss_pvc':
            loss_pvc=[0]
            loss_pvc[0]=trial.suggest_categorical('loss_pvc', choices=["L1", "L2"])
            params['loss_pvc']=loss_pvc

    error=train_and_eval(params=params)


    return error


def train_and_eval(params):
    params['output_folder']="none"
    params['ref']="none"
    params['output_pth']="none"
    verbose=params['verbose']

    train_dataloader, test_dataloader, validation_dataloader, params = dataset.load_data(params)

    DeepPVEModel = Model_instance.ModelInstance(params=params, from_pth=None, resume_training=False)
    device = DeepPVEModel.device


    for epoch in range(1,DeepPVEModel.n_epochs+1):
        if verbose>0:
            print("Epoch {}/{}".format(epoch, DeepPVEModel.n_epochs))

        for step,(batch_inputs,batch_targets) in enumerate(train_dataloader):
            for key_inputs in batch_inputs.keys():
                batch_inputs[key_inputs] = batch_inputs[key_inputs].to(device, non_blocking=True)
            for key_targets in batch_targets.keys():
                batch_targets[key_targets] = batch_targets[key_targets].to(device, non_blocking=True)

            DeepPVEModel.input_data(batch_inputs=batch_inputs, batch_targets=batch_targets)
            DeepPVEModel.optimize_parameters()

        DeepPVEModel.update_epoch()


    MNRMSE, MNMAE = helpers_functions.validation_errors(validation_dataloader, DeepPVEModel,
                                                        do_NRMSE=(params['validation_norm'] == "L2"),
                                                        do_NMAE=(params['validation_norm'] == "L1"))

    if params["validation_norm"]=="L2":
        return MNRMSE.item()
    elif params["validation_norm"]=="L1":
        return MNMAE.item()


if __name__ == '__main__':
    host = os.uname()[1]
    if (host !='suillus'):
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
    optuna_study()


