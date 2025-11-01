#!/usr/bin/env python3

import argparse
import sys
import os
host = os.uname()[1]
if (host !='siullus'):
    sys.path.append("/linkhome/rech/gencre01/uyo34ub/homeMocamed/WORK/PVE/MIRTorch")
else:
    sys.path.append("/export/home/tkaprelian/Desktop/External_repositories/MIRTorch")
from mirtorch.linear.spect import SPECT

import torch
import math
import itk
import numpy as np

def projs_rtk_to_mir(projs):
    projs_ = np.zeros((projs.shape[0], projs.shape[2], projs.shape[1]), dtype = projs.dtype)
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

def mlem(x, p, SPECT_sys, niter):
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

def get_psf(kernel_size, sigma0, alpha, nview, ny,sy, sid):
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.

    psf = torch.zeros((kernel_size, kernel_size, ny,nview), dtype = torch.float32)

    if (alpha>0 and sigma0>0):
        for iv in range(nview):
            dist = ny * sy + (sid - ny * sy / 2)
            for iy in range(ny):
                if iy<ny-1:
                    sigma = dist * 2 * sy * (alpha)**2 + (2* sy * alpha * sigma0) - alpha**2 * sy**2
                else:
                    sigma = alpha * dist + sigma0

                variance = sigma ** 2.

                gaussian_kernel=(1. / (2. * math.pi * variance)) * torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )
                psf[:,:,iy,iv] =gaussian_kernel / torch.sum(gaussian_kernel)

                dist = dist - sy
    else:
        psf[kernel_size//2, kernel_size//2, :, :] = 1
    return psf

def main():
    print(args)

    dtype = torch.float32

    attmap = itk.imread(args.attmap)
    attmap_tensor = torch.from_numpy(itk.array_from_image(attmap)).to(dtype)
    attmap_tensor = attmap_tensor.permute(2,0,1)
    spacing = np.array(attmap.GetSpacing())
    spx,spy,spz = spacing[0], spacing[1], spacing[2]
    nx,ny,nz = attmap_tensor.shape[0], attmap_tensor.shape[1], attmap_tensor.shape[2]

    nprojs = 120
    dy = spy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    attmap_tensor = attmap_tensor.to(device)

    if args.rm:
        psf = get_psf(kernel_size=7,sigma0=1.1684338873367237,alpha=0.03235363042582603,nview=120,
                  ny=ny,sy=dy,sid = 280).to(device).to(dtype)


        A = SPECT(size_in=(nx, ny, nz), size_out=(256, 256, nprojs),
              mumap=attmap_tensor, psfs=psf, dy=dy)
    else:
        psf = get_psf(kernel_size=1,sigma0=0,alpha=0,nview=120,
                  ny=ny,sy=dy,sid=280).to(device).to(dtype)

        A = SPECT(size_in=(nx, ny, nz), size_out=(256, 256, nprojs),
              mumap=attmap_tensor, psfs=psf, dy=dy)


    input = itk.imread(args.input)
    input_tensor = torch.from_numpy(itk.array_from_image(input).astype(np.float32)).to(device)
    input_tensor = input_tensor.permute(2, 0, 1)

    output_noRM = A._apply(input_tensor)

    itk.imwrite(itk.image_from_array(projs_mir_to_rtk(output_noRM.detach().cpu().numpy())), args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--attmap")
    parser.add_argument("--output")
    parser.add_argument("--rm", action="store_true")
    args = parser.parse_args()

    main()

