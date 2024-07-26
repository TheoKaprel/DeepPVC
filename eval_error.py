
import torch
import time
import json as js
import os
import numpy as np
import click
from DeepPVC import dataset, Model_instance
from DeepPVC import helpers, helpers_params, helpers_functions, helpers_data

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth', 'lpth', multiple = True)
@click.option('--dataset','dataset_path', help='h5 dataset to eval pth and calc errors',required=True)
@click.option('--output_folder', '-f', help='Output folder ',required=True)
def train_onclick(lpth,dataset_path, output_folder):
    train(lpth=lpth,dataset_path=dataset_path,output_folder=output_folder)

def train(lpth,dataset_path,output_folder):
    device = helpers.get_auto_device("auto")

    for pth in lpth:

        pth_errors = []

        checkpoint = torch.load(pth, map_location=device)
        params = checkpoint['params']

        helpers_params.check_params(params)

        test_dataset = dataset.get_dataset(params=params, paths=[dataset_path],test=True)
        test_batchsize = 1
        if on_jz:
            params['jean_zay'] = True
            import idr_torch
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,
                                                                      num_replicas=idr_torch.size,
                                                                      rank=idr_torch.rank,
                                                                      shuffle=False)
        else:
            params['jean_zay'] = False
            test_sampler = None

        test_dataloader = dataset.DataLoader(dataset=test_dataset,
                                      batch_size=test_batchsize,
                                      shuffle=False,
                                      num_workers=params['num_workers'],
                                      pin_memory=True,
                                      sampler=test_sampler)



        DeepPVEModel = Model_instance.ModelInstance(params=params, from_pth=pth,resume_training=False)
        DeepPVEModel.load_model(pth_path=pth, new_lr=None)

        if rank == 0:
            DeepPVEModel.show_infos()

        device = DeepPVEModel.device

        DeepPVEModel.switch_eval()

        for step, (batch_inputs, batch_targets) in enumerate(test_dataloader):

            for key_inputs in batch_inputs.keys():
                batch_inputs[key_inputs] = batch_inputs[key_inputs].to(device, non_blocking=True)
            for key_targets in batch_targets.keys():
                batch_targets[key_targets] = batch_targets[key_targets].to(device, non_blocking=True)

            with torch.no_grad():
                fakePVfree = DeepPVEModel.forward(batch_inputs)

            ground_truth=batch_targets['PVfree'] if (DeepPVEModel.params['inputs'] == "full_sino") else batch_targets['src_4mm']
            NRMSE_batch = torch.sqrt(torch.mean((fakePVfree - ground_truth) ** 2)) / torch.sqrt(torch.mean(ground_truth**2))
            pth_errors.append(NRMSE_batch.item())

        pth_errors = np.array(pth_errors)
        np.save(os.path.join(output_folder,(pth.split('/')[-1]).replace('.pth', '.npy')),pth_errors)





if __name__ == '__main__':
    host = os.uname()[1]
    if (host != 'siullus'):
        import idr_torch

        on_jz = True
        # get distributed configuration from Slurm environment
        NODE_ID = os.environ['SLURM_NODEID']
        MASTER_ADDR = os.environ['MASTER_ADDR'] if ("MASTER_ADDR" in os.environ) else os.environ['HOSTNAME']
        TIME_LIMIT = os.environ['SBATCH_TIMELIMIT']

        print(f"TIME_LIMIT is : {TIME_LIMIT}")
        TIME_LIMIT_s = 0
        TIME_LIMIT_split = TIME_LIMIT.split(":")
        if len(TIME_LIMIT_split) == 3:
            TIME_LIMIT_s = int(TIME_LIMIT_split[0]) * 60 * 60 + int(TIME_LIMIT_split[1]) * 60 + int(TIME_LIMIT_split[2])
        elif len(TIME_LIMIT_split) == 2:
            TIME_LIMIT_s = int(TIME_LIMIT_split[0]) * 60 + int(TIME_LIMIT_split[1])
        print(f"i.e. {TIME_LIMIT_s} seconds")

        # display info
        if idr_torch.rank == 0:
            print(">>> Training on ", len(idr_torch.hostnames), " nodes and ", idr_torch.size,
                  " processes, master node is ", MASTER_ADDR)
        print("- Process {} corresponds to GPU {} of node {}".format(idr_torch.rank, idr_torch.local_rank, NODE_ID))

        dist.init_process_group(backend='nccl', init_method='env://', world_size=idr_torch.size, rank=idr_torch.rank)

        rank = idr_torch.rank
    else:
        rank = 0
        on_jz = False

    train_onclick()


