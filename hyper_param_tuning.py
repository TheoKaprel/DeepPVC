#!/usr/bin/env python3
import torch.distributed

from DeepPVC import dataset,Models,helpers_data,helpers_functions

import click
import optuna
import json
import os

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--n_trials", type=int, default=30)
@click.option("--parambase")
def optuna_study(n_trials, parambase):
    params_file = open(parambase).read()
    print(params_file)
    params = json.loads(params_file)
    objective = lambda trial: objective_w_params(trial=trial,params=params)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    print('*'*50)
    print('Final Result : ')
    for key, value in best_trial.params.items():
        print("{}: {}\n".format(key, value))


def objective_w_params(trial, params):

    params['learning_rate']=trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

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
    if params['jean_zay'] and idr_torch.rank==0:
        torch.distributed.destroy_process_group()


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


    optuna_study()


