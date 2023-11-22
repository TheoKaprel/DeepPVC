#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import torch
from DeepPVC import helpers,dataset,helpers_params,helpers_data,helpers_functions,Model_instance
from torch.utils.data import DataLoader

def main():
    print(args)
    device = helpers.get_auto_device("cuda")
    dict_mse,dict_mae= {},{}
    for pth in args.pth:
        print('----------------')
        nn = torch.load(pth,map_location=device)
        params = nn['params']
        # helpers_params.check_params(params)
        params['jean_zay']=False
        ref=params['ref']
        params['verbose']=0
        print(ref)
        eval_batchsize = params['test_batchsize']//4
        eval_dataset = dataset.get_dataset(params=params,paths=[args.dataset],test=True)
        eval_dataloader = DataLoader(dataset=eval_dataset,
                                     batch_size=eval_batchsize,
                                     shuffle=False,
                                     num_workers=params['num_workers'],
                                     pin_memory=True,
                                     sampler=None)


        model = Model_instance.ModelInstance(params=params, from_pth=pth,resume_training=False, device=device)
        model.load_model(pth_path=pth)
        model.switch_device(device)
        model.switch_eval()

        MSE, MAE = helpers_functions.validation_errors(eval_dataloader, model,
                                                       do_NRMSE=args.mse,
                                                       do_NMAE=args.mae)
        dict_mse[ref]=MSE.item()
        dict_mae[ref]=MAE.item()

    fig,ax = plt.subplots(1,2)
    ax[0].bar(dict_mse.keys(),dict_mse.values())
    ax[1].bar(dict_mae.keys(),dict_mae.values())
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", nargs='+')
    parser.add_argument("--dataset")
    parser.add_argument("--mse", action="store_true")
    parser.add_argument("--mae", action="store_true")
    args = parser.parse_args()

    main()
