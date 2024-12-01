#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import torch
import numpy as np
import itk

from DeepPVC import Model_instance, helpers_data, helpers, helpers_params

import sys
import os
host = os.uname()[1]
if (host !='suillus'):
    sys.path.append("/linkhome/rech/gencre01/uyo34ub/homeMocamed/WORK/PVE/MIRTorch")
else:
    sys.path.append("/export/home/tkaprelian/Desktop/External_repositories/MIRTorch")

from mirtorch.linear.spect import SPECT
from mirtorch_mlem_recons import projs_rtk_to_mir,projs_mir_to_rtk,osem,get_psf, get_psf_
import matplotlib.pyplot as plt

def main():
    print(args)
    input_image = itk.imread(args.input)
    vSpacing = np.array(input_image.GetSpacing())
    vOffset = np.array(input_image.GetOrigin())

    device = helpers.get_auto_device(device_mode="auto")
    pth_file = torch.load(args.pth, map_location=device)
    params = pth_file['params']
    helpers_params.check_params(params)

    params['jean_zay']=False
    model = Model_instance.ModelInstance(params=params, from_pth=args.pth,resume_training=False,device=device)
    model.load_model(pth_path=args.pth)
    model.switch_device(device)
    model.switch_train()
    model.show_infos()

    input_PVE_noisy_array = itk.array_from_image(itk.imread(args.input))
    input_rec_fp_array = itk.array_from_image(itk.imread(args.input_rec_fp)) if ((args.input_rec_fp is not None) and (params['with_rec_fp'] or params['with_PVCNet_rec'])) else None
    attmap_fp_array = itk.array_from_image(itk.imread(args.attmap)) if (args.attmap is not None) else None


    data_input = helpers_data.get_dataset_for_eval(params=params,
                                                   input_PVE_noisy_array=input_PVE_noisy_array,
                                                   input_rec_fp_array=input_rec_fp_array,
                                                   attmap_fp_array=attmap_fp_array)

    for key in data_input.keys():
        data_input[key] = data_input[key].to(device)

    projs = itk.imread(args.projs)
    projs_array = itk.array_from_image(projs).astype(np.float32)
    projs_array_mir = projs_rtk_to_mir(projs_array)
    projs_tensor_mir = torch.from_numpy(projs_array_mir).to(device)


    attmap_tensor = torch.from_numpy(attmap_fp_array.astype(np.float32))
    attmap_tensor_mirt = attmap_tensor.transpose(0,1).transpose(0,2).to(device)
    spacing = np.array(input_image.GetSpacing())
    spx,spy,spz = spacing[0], spacing[1], spacing[2]
    nx,ny,nz = attmap_tensor.shape[0], attmap_tensor.shape[1], attmap_tensor.shape[2]
    nprojs = 120
    dy = spy

    kernel_size = 7
    psf_RM = get_psf(kernel_size=kernel_size,sigma0=1.1684338873367237,alpha=0.03235363042582603,nview=120,
                  ny=ny,sy=dy,sid = 280).to(device) # (7, 7, 128, 120)

    A_RM = SPECT(size_in=(nx, ny, nz), size_out=(128, 128, nprojs),
              mumap=attmap_tensor_mirt, psfs=psf_RM, dy=dy,first_angle=0)
    print(f"PSF shape: {psf_RM.shape}")

    poisson_loss = torch.nn.PoissonNLLLoss(log_input=False, eps=1e-4, reduction="mean")
    optimizer = torch.optim.Adam(model.UNet.parameters(), lr=args.lr)
    model.set_requires_grad(model.UNet,requires_grad=True)

    num_epochs = args.nepochs

    for epoch in range(1, num_epochs+1):
        output=model.forward(data_input)[0,:,:,:]*4.7952
        output_mirt = output.transpose(0,1).transpose(0,2)
        fp_img = A_RM._apply(output_mirt)
        loss_k = poisson_loss(fp_img,projs_tensor_mir)

        # Backward pass
        optimizer.zero_grad()  # Clear previous gradients
        loss_k.backward()  # Compute gradients
        optimizer.step()  # Update model parameters
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss_k.item():.6f}")

        if epoch % args.saveevery==0:
            itk.imwrite(itk.image_from_array(output.detach().cpu().numpy()),os.path.join(args.savefolder, f"iter_{epoch}.mhd"))
            # itk.imwrite(itk.image_from_array(fp_img_1.detach().cpu().numpy()),os.path.join(args.savefolder, f"fp1_img_{epoch}.mhd"))
            # itk.imwrite(itk.image_from_array(projs_tensor_mir.detach().cpu().numpy()),os.path.join(args.savefolder, f"projs_tensor_mir.mhd"))




    # output_array = helpers_data.back_to_input_format(params=params,output=output, initial_shape = list(input_PVE_noisy_array.shape))
    # output_image = itk.image_from_array(output_array)
    # output_image.SetSpacing(vSpacing)
    # output_image.SetOrigin(vOffset)
    # itk.imwrite(output_image, args.output)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth")
    parser.add_argument("--input")
    parser.add_argument("--attmap")
    parser.add_argument("--input_rec_fp")
    parser.add_argument("--projs")
    parser.add_argument("--output")
    parser.add_argument("--nepochs", type = int)
    parser.add_argument("--saveevery", type = int)
    parser.add_argument("--savefolder")
    parser.add_argument("--lr", type = float)
    args = parser.parse_args()

    main()
