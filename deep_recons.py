#!/usr/bin/env python3

import argparse
import numpy as np
import itk
from itk import RTK as rtk

import torch
from torch import optim
import torch.nn as nn

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from PVE_data.Analytical_data.parameters import get_psf_params

from DeepPVC.networks import UNet


import pytomography
from pytomography.transforms import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM,FilteredBackProjection
from pytomography.projectors import SPECTSystemMatrix
from pytomography.metadata.SPECT import SPECTPSFMeta,SPECTProjMeta,SPECTObjectMeta


def get_system_matrix(projs, attmap,sigma, alpha, sid, nprojs):
    print("--------projs meta data-------")
    projs_array=itk.array_from_image(projs).astype(float)
    shape_proj = projs_array.shape
    projs_spacing = np.array(projs.GetSpacing())
    print(projs_spacing)
    dx = projs_spacing[0] / 10
    dz = projs_spacing[1] / 10
    dr = (dx, dx, dz)
    angles = np.linspace(0, 360, nprojs+1)[:nprojs]
    radii = sid/10 * np.ones_like(angles)
    proj_meta = SPECTProjMeta((shape_proj[1], shape_proj[2]), angles, radii)
    proj_meta.filepath = "none"
    proj_meta.index_peak = 0

    print("--------object meta data-------")
    shape_obj = (shape_proj[1], shape_proj[1], shape_proj[2])
    object_meta = SPECTObjectMeta(dr, shape_obj)
    M = np.zeros((4,4))
    M[0] = np.array([dx, 0, 0, 0])
    M[1] = np.array([0, dx, 0, 0])
    M[2] = np.array([0, 0, -dz, 0])
    M[3] = np.array([0, 0, 0, 1])
    object_meta.affine_matrix = M

    print("-------- get projections -------")
    projections = np.transpose(projs_array[:,::-1,:], (0,2,1)).astype(np.float32)
    photopeak= torch.tensor(projections.copy()).to(pytomography.dtype).to(pytomography.device)
    photopeak = photopeak.unsqueeze(dim=0)

    print("-------- attenuation -------")
    attmap_array = itk.array_from_image(attmap)
    print(attmap_array.shape)
    attmap_array_t = np.transpose(attmap_array[:,::-1,:], (2,0,1)).astype(np.float32)
    t = torch.from_numpy(attmap_array_t)

    tpadded = torch.nn.functional.pad(t, (
    (shape_proj[1] - t.shape[2]) // 2, (shape_proj[1] - t.shape[2]) // 2 + (shape_proj[1] - t.shape[2]) % 2,
    (shape_proj[1] - t.shape[1]) // 2, (shape_proj[1] - t.shape[1]) // 2 + (shape_proj[1] - t.shape[1]) % 2,
    (shape_proj[1] - t.shape[0]) // 2, (shape_proj[1] - t.shape[0]) // 2 + (shape_proj[1] - t.shape[0]) % 2))

    psf_meta = SPECTPSFMeta((alpha, sigma))
    print('PSF META')
    psf_transform = SPECTPSFTransform(psf_meta)
    att_transform = SPECTAttenuationTransform(attenuation_map=tpadded)

    system_matrix = SPECTSystemMatrix(
        obj2obj_transforms=[att_transform,psf_transform],
        proj2proj_transforms=[],
        object_meta=object_meta,
        proj_meta=proj_meta)

    return system_matrix, photopeak


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        sequence = []

        list_channels = [1, 8, 8, 8, 8, 8, 8]

        for k in range(len(list_channels)-1):
            sequence.append(nn.Conv3d(in_channels=list_channels[k], out_channels=list_channels[k+1],
                                           kernel_size=(3,3,3),stride=(1,1,1),padding=1))
            sequence.append(nn.ReLU(inplace=True))

        sequence.append(nn.Conv3d(in_channels=list_channels[-1], out_channels=1,
                                  kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1))

        self.sequenceCNN = nn.Sequential(*sequence)
        self.activation= nn.ReLU(inplace=True)

    def forward(self,x):
        res = x
        y = self.sequenceCNN(x)
        y = y + res
        return self.activation(y)

def main():
    print(args)
    #-------------------------------------------------#
    niter = args.n
    sid = args.sid
    nprojs = args.nprojs
    #-------------------------------------------------#
    # type
    pixelType = itk.F
    imageType = itk.Image[pixelType, 3]
    #-------------------------------------------------#
    # projs
    projs_itk = itk.imread(args.projs, pixel_type=pixelType)
    projs_array = itk.array_from_image(projs_itk)
    #-------------------------------------------------#
    # attmap
    attmap_itk = itk.imread(args.attmap, pixel_type=pixelType)
    # -------------------------------------------------#
    # psf
    sigma0_psf, alpha_psf, _ = get_psf_params(machine=args.spect_system)
    # -------------------------------------------------#
    # matrices
    matrix_no_RM,_ = get_system_matrix(projs=projs_itk, attmap=attmap_itk, sigma=0, alpha=0, sid=sid, nprojs=nprojs)
    matrix_RM,photopeak = get_system_matrix(projs=projs_itk, attmap=attmap_itk, sigma=sigma0_psf/10, alpha=alpha_psf, sid=sid, nprojs=nprojs)

    reconstruction_algorithm = OSEM(
        projections=photopeak,
        system_matrix=matrix_RM)
    reconstructed_object = reconstruction_algorithm(n_iters=niter, n_subsets=15)
    itk.imwrite(itk.image_from_array(reconstructed_object.cpu().numpy()),args.output)
    reconstructed_object_fp = matrix_no_RM.forward(reconstructed_object)
    print("projections after forward : ")
    print(reconstructed_object_fp.shape)
    itk.imwrite(itk.image_from_array(reconstructed_object_fp.squeeze().cpu().numpy()),args.output.replace('.mhd', '_fp.mhd'))

    # -------------------------------------------------#
    # model
    device = torch.device("cuda")
    # unet = UNet(input_channel=1,ngc=2,init_feature_kernel=3,paths=False,output_channel=1,nb_ed_layers=3,
    #             generator_activation="relu", use_dropout=True,leaky_relu=0.2,norm="inst_norm",
    #             residual_layer=0,blocks=("conv-norm-relu-pool", "convT-norm-relu"),ResUnet=False,
    #             dim=3,final_2dchannels=False, final_2dconv=0).to(device=device)

    unet = CNN().to(device=device)
    print(unet)
    nb_params = sum(p.numel() for p in unet.parameters())
    print(f'NUMBER OF PARAMERS : {nb_params}')

    #-------------------------------------------------#
    # loss,optimizer
    optimizer = optim.Adam(unet.parameters(), lr=0.001)
    loss = torch.nn.L1Loss()

    print('req grad : ')
    print(photopeak.requires_grad)
    photopeak.requires_grad_(True)
    print(photopeak.requires_grad)

    for k in range(niter):
        print(k)
        # projs_corrected = h(projs)
        # recons_corrected = BP(projs_corrected)
        # recons_corrected_fp = FP_rm(recons_corrected)
        # loss = loss(projs, recons_corrected_fp)
        # update h

        projs_corrected = unet(photopeak[None,:,:,:,:])[0,:,:,:,:]
        recons_corrected = matrix_no_RM.backward(projs_corrected)
        recons_corrected_fp = matrix_RM.forward(recons_corrected)
        loss_k = loss(recons_corrected_fp, photopeak)

        (loss_k).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"loss {k} : {loss_k:.5f}")


    itk.imwrite(itk.image_from_array(recons_corrected.squeeze().cpu().numpy()),args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int)
    parser.add_argument("--projs")
    parser.add_argument("--nprojs", type=int)
    parser.add_argument("--sid", type=float)
    parser.add_argument("--attmap")
    parser.add_argument("--spect_system")
    parser.add_argument("--output")
    args = parser.parse_args()

    main()
