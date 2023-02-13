#!/usr/bin/env python3
import torch.distributed as dist


from DeepPVC import dataset,Models,helpers_data,helpers_functions

import click
import optuna
import json
import os

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--n_trials", type=int, default=30)
@click.option("--parambase")
@click.option("--tune", multiple=True)
def optuna_study(n_trials, parambase, tune):
    params_file = open(parambase).read()
    params = json.loads(params_file)
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
        if param_to_tune=='learning_rate':
            params['learning_rate']=trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        elif param_to_tune=="lr_mult_rate":
            policy=["multiplicative",1]
            policy[1]=trial.suggest_float('lr_mult_rate', 0.9, 1)
        elif param_to_tune=='conv3d':
            params['conv3d']=trial.suggest_categorical('conv3d', choices=[True, False])
        elif param_to_tune == 'residual_layer':
            params['residual_layer'] = trial.suggest_categorical('residual_layer', choices=[True, False])
        elif param_to_tune=="init_feature_kernel":
            params['init_feature_kernel']=trial.suggest_discrete_uniform('init_feature_kernel', low=3,high=9,q=2)
        elif param_to_tune=="nb_ed_layers":
            params['nb_ed_layers']=trial.suggest_int('nb_ed_layers',3,5)
        elif param_to_tune=='hidden_channels_gen':
            params['hidden_channels_gen']=trial.suggest_categorical('hidden_channels_gen', choices=[16,32,64])
        elif param_to_tune == 'hidden_channels_disc':
            params['hidden_channels_disc'] = trial.suggest_categorical('hidden_channels_disc', choices=[7, 9, 11,13])
        elif param_to_tune=='lambda_recon':
            lambda_recon=[0]
            lambda_recon[0]=trial.suggest_int('lambda_recon', 50,150)
            params['lambda_recon']=lambda_recon
        else:
            print('ERROR: unrecognized parameter to tune {}'.format(param_to_tune))


    error=train_and_eval(params=params)


    return error


def train_and_eval(params):
    params['output_folder']="none"
    params['ref']="none"
    params['output_pth']="none"
    verbose=params['verbose']

    train_dataloader, test_dataloader, params = dataset.load_data(params)

    DeepPVEModel = Models.ModelInstance(params=params, from_pth=None, resume_training=False)
    device = DeepPVEModel.device
    data_normalisation = params['data_normalisation']


    for epoch in range(1,DeepPVEModel.n_epochs+1):
        if verbose>0:
            print("Epoch {}/{}".format(epoch, DeepPVEModel.n_epochs))

        for step,batch in enumerate(train_dataloader):
            norm = helpers_data.compute_norm_eval(dataset_or_img=batch,data_normalisation=data_normalisation)
            batch = helpers_data.normalize_eval(dataset_or_img=batch,data_normalisation=data_normalisation,norm=norm,params=params,to_torch=False)

            batch = batch.to(device,non_blocking=True)
            DeepPVEModel.input_data(batch)

            DeepPVEModel.optimize_parameters()

        DeepPVEModel.update_epoch()


    MNRMSE, MNMAE = helpers_functions.validation_errors(test_dataloader, DeepPVEModel,
                                                        do_NRMSE=(params['validation_norm'] == "L2"),
                                                        do_NMAE=(params['validation_norm'] == "L1"))

    if params["validation_norm"]=="L2":
        return MNRMSE.item()
    elif params["validation_norm"]=="L1":
        return MNMAE.item()


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

        dist.init_process_group(backend='nccl', init_method='env://', world_size=idr_torch.size, rank=idr_torch.rank)
        rank=idr_torch.rank
    else:
        rank=0
    optuna_study()


