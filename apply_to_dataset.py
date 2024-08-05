
import torch
import os
import numpy as np
import click
from DeepPVC import dataset, Model_instance
from DeepPVC import helpers, helpers_params
import torch.distributed as dist

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth', 'lpth', multiple = True)
@click.option('--dataset','dataset_path', help='h5 dataset to eval pth and calc errors',required=True)
@click.option('--output_folder', '-f', help='Output folder ',required=True)
def applyonclick(lpth,dataset_path, output_folder):
    apply(lpth=lpth,dataset_path=dataset_path,output_folder=output_folder)

def apply(lpth,dataset_path,output_folder):
    device = helpers.get_auto_device("auto")

    for pth in lpth:

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
                if key_inputs!="ref":
                    batch_inputs[key_inputs] = batch_inputs[key_inputs].to(device, non_blocking=True)
            for key_targets in batch_targets.keys():
                batch_targets[key_targets] = batch_targets[key_targets].to(device, non_blocking=True)

            with torch.no_grad():
                fakePVfree = DeepPVEModel.forward(batch_inputs)[0,4:124,:,:]
            print(fakePVfree.shape)
            ref = str(batch_inputs['ref'][0], "utf-8")
            output_fn = f"{ref}_PVCNet_"+DeepPVEModel.output_pth.replace(".pth",".npy")
            np.save(os.path.join(output_folder,output_fn), fakePVfree.detach().cpu().numpy())



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

    applyonclick()


