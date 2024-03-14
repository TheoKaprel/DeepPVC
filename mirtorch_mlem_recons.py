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
from torch import optim
import torch.nn as nn

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

def deep_mlem(p, SPECT_sys_noRM, SPECT_sys_RM, niter, net, loss, optimizer):
    # projs_corrected = h(projs)
    # recons_corrected = BP(projs_corrected)
    # recons_corrected_fp = FP_rm(recons_corrected)
    # loss = loss(projs, recons_corrected_fp)
    # update h
    for k in range(niter):
        p_hat = net(p[None,None,:,:,:].float())[0,0,:,:,:]
        rec_corrected = SPECT_sys_noRM._apply_adjoint(p_hat)
        rec_corrected_fp = SPECT_sys_RM._apply(rec_corrected)
        loss_k = loss(p, rec_corrected_fp)
        loss_k.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"loss {k} : {loss_k}")

    p_hat = net(p)
    out = SPECT_sys_noRM._apply_adjoint(p_hat)
    return out

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

    projs = itk.imread(args.projs)
    projs_array = itk.array_from_image(projs).astype(np.float32)
    projs_array_mir = projs_rtk_to_mir(projs_array)
    projs_tensor_mir = torch.from_numpy(projs_array_mir)


    attmap = itk.imread(args.attmap)
    attmap_tensor = torch.from_numpy(itk.array_from_image(attmap).astype(np.float32))
    attmap_tensor = attmap_tensor.permute(2,0,1)
    spacing = np.array(attmap.GetSpacing())
    spx,spy,spz = spacing[0], spacing[1], spacing[2]
    nx,ny,nz = attmap_tensor.shape[0], attmap_tensor.shape[1], attmap_tensor.shape[2]

    nprojs = 120
    dy = spy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    attmap_tensor = attmap_tensor.to(device)


    kernel_size = 3
    psf = torch.zeros((kernel_size, kernel_size, ny,nprojs), dtype = torch.float32).to(device)
    psf[1,1,:,:]=1
    psf = psf.to(device)

    A = SPECT(size_in=(nx, ny, nz), size_out=(256, 256, nprojs),
              mumap=attmap_tensor, psfs=psf, dy=dy)

    x0 = torch.ones(nx, ny, nz) # initial uniform image
    # MLEM reconstruction after 20 iterations
    print(f"p shape : {projs_tensor_mir.shape}")

    x0 = x0.to(device)
    projs_tensor_mir = projs_tensor_mir.to(device)

    unet = CNN().to(device=device)
    print(unet)
    nb_params = sum(p.numel() for p in unet.parameters())
    print(f'NUMBER OF PARAMERS : {nb_params}')
    # loss,optimizer
    optimizer = optim.Adam(unet.parameters(), lr=args.lr)
    loss = torch.nn.L1Loss()


    # xn = mlem(x=x0,p=projs_tensor_mir,SPECT_sys=A,niter=args.niter, net = unet)
    xn = deep_mlem(p = projs_tensor_mir,
                   SPECT_sys_RM=A, SPECT_sys_noRM=A,
                   niter=args.niter,net=unet,
                   loss = loss,optimizer=optimizer)

    rec_array = xn.detach().cpu().numpy()
    rec_array_ = np.transpose(rec_array, (1,2,0))
    rec_itk  =itk.image_from_array(rec_array_)
    itk.imwrite(rec_itk, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--projs")
    parser.add_argument("--attmap")
    parser.add_argument("--niter", type =int)
    parser.add_argument("--lr", type =float, default=0.001)
    parser.add_argument("--output")
    args = parser.parse_args()

    main()
