#!/usr/bin/env python3

import argparse
import sys
sys.path.append("~/Desktop/External_repositories/MIRTorch")
from mirtorch.linear.spect import SPECT
import torch
import itk
import numpy as np

def projs_rtk_to_mir(projs):
    projs_ = np.zeros((projs.shape[0], projs.shape[2], projs.shape[1]))
    for k in range(projs.shape[0]):
        projs_[k,:,:] = projs[k,:,:].transpose()
    projs_ = np.transpose(projs_, (1, 2, 0))
    return projs_

def projs_mir_to_rtk(projs):
    projs = np.transpose(projs, (2,0,1))
    projs_ = np.zeros((projs.shape[0], projs.shape[2], projs.shape[1]))
    for k in range(projs.shape[0]):
        projs_[k,:,:] = projs[k,:,:].transpose()
    return projs_

def mlem(x, p, SPECT_sys, niter=20):
    asum = SPECT_sys._apply_adjoint(torch.ones_like(p))
    asum[asum == 0] = float('Inf')
    out = torch.clone(x)
    for iter in range(niter):
        print(f'iter : {iter}')
        ybar = SPECT_sys._apply(out)
        yratio = torch.div(p, ybar)
        back = SPECT_sys._apply_adjoint(yratio)
        out = torch.multiply(out, torch.div(back, asum))
    return out


def main():
    print(args)

    projs = itk.imread(args.projs)
    projs_array = itk.array_from_image(projs).astype(np.float16)
    projs_array_mir = projs_rtk_to_mir(projs_array)
    projs_tensor_mir = torch.from_numpy(projs_array_mir)


    attmap = itk.imread(args.attmap)
    attmap_tensor = torch.from_numpy(itk.array_from_image(attmap).astype(np.float16))
    attmap_tensor = attmap_tensor.permute(2,0,1)
    spacing = np.array(attmap.GetSpacing())
    spx,spy,spz = spacing[0], spacing[1], spacing[2]
    nx,ny,nz = attmap_tensor.shape[0], attmap_tensor.shape[1], attmap_tensor.shape[2]

    nprojs = 120
    dy = spy

    kernel_size = 3
    psf = torch.zeros((kernel_size, kernel_size, ny,nprojs), dtype = torch.float16)
    psf[1,1,:,:]=1

    A = SPECT(size_in=(nx, ny, nz), size_out=(256, 256, nprojs),
              mumap=attmap_tensor, psfs=psf, dy=dy)

    x0 = torch.ones(nx, ny, nz) # initial uniform image
    # MLEM reconstruction after 20 iterations
    print(f"p shape : {projs_tensor_mir.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    x0 = x0.to(device)
    projs_tensor_mir = projs_tensor_mir.to(device)

    xn = mlem(x=x0,p=projs_tensor_mir,SPECT_sys=A,niter=args.niter)

    rec_array = xn.detach().cpu().numpy()
    rec_array_ = np.transpose(rec_array, (1,2,0))
    rec_itk  =itk.image_from_array(rec_array_)
    itk.imwrite(rec_itk, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--projs")
    parser.add_argument("--attmap")
    parser.add_argument("--niter", type =int)
    parser.add_argument("--output")
    args = parser.parse_args()

    main()
