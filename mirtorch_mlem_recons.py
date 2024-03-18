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

def deep_mlem(p, SPECT_sys_noRM, SPECT_sys_RM, niter, net, loss, optimizer):
    # projs_corrected = h(projs)
    # recons_corrected = BP(projs_corrected)
    # recons_corrected_fp = FP_rm(recons_corrected)
    # loss = loss(projs, recons_corrected_fp)
    # update h

    if loss.__class__==torch.nn.L1Loss:
        p_max = p.max()
        p = p / p_max
        norm = "max"
    elif ((loss.__class__==torch.nn.KLDivLoss) or (loss.__class__==torch.nn.PoissonNLLLoss)):
        norm = "log"

    print(f"NORM : {norm}")

    for k in range(niter):
        p_hatn = net(p[None,None,:,:,:])[0,0,:,:,:]
        p_hat = p_hatn * p_max if (norm=="max") else p_hatn
        rec_corrected = SPECT_sys_noRM._apply_adjoint(p_hat)
        rec_corrected_fp = SPECT_sys_RM._apply(rec_corrected)
        rec_corrected_fpn = (rec_corrected_fp / p_max) if (norm=="max") else torch.log(rec_corrected_fp+1e-8)
        loss_k = loss(rec_corrected_fpn, p)
        loss_k.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"loss {k} : {loss_k}")
        del rec_corrected,rec_corrected_fp
        itk.imwrite(itk.image_from_array(projs_mir_to_rtk(p_hat.detach().cpu().numpy())), os.path.join(args.iter, f"iter_{k}.mhd"))


    p_hat = net(p[None,None,:,:,:])[0,0,:,:,:] * p_max if norm=="max" else net(p[None,None,:,:,:])[0,0,:,:,:]
    out = SPECT_sys_noRM._apply_adjoint(p_hat)
    return out

def deep_mlem_v2(p, SPECT_sys_noRM, SPECT_sys_RM, niter, net, loss, optimizer):
    asum = SPECT_sys_noRM._apply_adjoint(torch.ones_like(p))
    asum[asum == 0] = float('Inf')
    out = torch.ones_like(asum)

    for iter in range(niter):
        print(f'iter : {iter}')
        optimizer.zero_grad(set_to_none=True)

        outhat = net(out[None,None,:,:,:])[0,0,:,:,:]

        ybar = SPECT_sys_RM._apply(outhat)

        loss_k = loss(ybar, p)
        loss_k.backward()
        optimizer.step()
        print(f"loss {iter} : {loss_k}")

        with torch.no_grad():
            yratio = torch.div(p, ybar)
            back = SPECT_sys_noRM._apply_adjoint(yratio)
            out = torch.multiply(out, torch.div(back, asum))
            # outk = out.clone()
            # itk.imwrite(itk.image_from_array((outk.detach().cpu().numpy())),
            #             os.path.join(args.iter, f"iter_{iter}.mhd"))

    return out

def deep_mlem_v3(p, SPECT_sys_noRM, SPECT_sys_RM, niter, net, loss, optimizer):
    # projs_corrected = h(projs)
    # recons_corrected = rec_no_rm(projs_corrected)
    # recons_corrected_fp = FP_rm(recons_corrected)
    # loss = loss(projs, recons_corrected_fp)
    # update h

    if loss.__class__==torch.nn.L1Loss:
        p_max = p.max()
        p = p / p_max
        norm = "max"
    elif ((loss.__class__==torch.nn.KLDivLoss) or (loss.__class__==torch.nn.PoissonNLLLoss)):
        norm = "log"

    print(f"NORM : {norm}")

    for k in range(niter):
        p_hatn = net(p[None,None,:,:,:])[0,0,:,:,:]
        p_hat = p_hatn * p_max if (norm=="max") else p_hatn
        del p_hatn
        asum = SPECT_sys_noRM._apply_adjoint(torch.ones_like(p))
        asum[asum == 0] = float('Inf')
        rec_corrected = torch.ones_like(asum)
        for _ in range(5):
            print(f"    inner {_}")
            ybar = SPECT_sys_noRM._apply(rec_corrected)
            ybar[ybar == 0] = 1
            yratio = torch.div(p_hat, ybar)
            del ybar
            back = SPECT_sys_noRM._apply_adjoint(yratio)
            del yratio
            rec_corrected = torch.multiply(rec_corrected, torch.div(back, asum))
            del back

        del asum

        rec_corrected_fp = SPECT_sys_RM._apply(rec_corrected)
        rec_corrected_fpn = (rec_corrected_fp / p_max) if (norm=="max") else torch.log(rec_corrected_fp+1e-8)
        loss_k = loss(rec_corrected_fpn, p)
        loss_k.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"loss {k} : {loss_k}")
        del rec_corrected,rec_corrected_fp
        itk.imwrite(itk.image_from_array(projs_mir_to_rtk(p_hat.detach().cpu().numpy())), os.path.join(args.iter, f"iter_{k}.mhd"))


    p_hat = net(p[None,None,:,:,:])[0,0,:,:,:] * p_max if norm=="max" else net(p[None,None,:,:,:])[0,0,:,:,:]
    out = SPECT_sys_noRM._apply_adjoint(p_hat)
    return out


def deep_mlem_v4(p, SPECT_sys_RM, niter, net, loss, optimizer):
    asum = SPECT_sys_RM._apply_adjoint(torch.ones_like(p))
    asum[asum == 0] = float('Inf')
    out = torch.ones_like(asum)

    for iter in range(niter):
        out_hat = net(out[None, None, :, :, :])[0, 0, :, :, :]

        ybar = SPECT_sys_RM._apply(out_hat)
        loss_k = loss(ybar, p)
        loss_k.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"loss {iter} : {loss_k}")

        with torch.no_grad():
            ybar[ybar == 0] = float('Inf')

            yratio = torch.div(p, ybar)
            back = SPECT_sys_RM._apply_adjoint(yratio)
            out = torch.multiply(out_hat, torch.div(back, asum))
            itk.imwrite(itk.image_from_array((out_hat.detach().cpu().numpy())), os.path.join(args.iter, f"iter_{iter}.mhd"))

    out_hat = net(out[None, None, :, :, :])[0, 0, :, :, :]

    return out_hat


def deep_mlem_v5(p, SPECT_sys_RM, niter, net, loss, optimizer):
    print('OSEM-RM')
    with torch.no_grad():
        x0 = torch.ones_like(SPECT_sys_RM.mumap)
        x_RM = mlem(x=x0, p=p, SPECT_sys=SPECT_sys_RM, niter=20)

    print("Training")
    for iter in range(niter):
        out_hat = net(x_RM[None, None, :, :, :])[0, 0, :, :, :]

        ybar = SPECT_sys_RM._apply(out_hat)
        loss_k = loss(ybar, p)
        loss_k.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"loss {iter} : {loss_k}")
        itk.imwrite(itk.image_from_array((out_hat.detach().cpu().numpy())), os.path.join(args.iter, f"iter_{iter}.mhd"))

    return out_hat



class CNN(nn.Module):
    def __init__(self, nc=8, ks = 3, nl = 6):
        super(CNN, self).__init__()
        sequence = []

        list_channels = [1]
        for _ in range(nl):
            list_channels.append(nc)

        p = (ks-1)//2

        for k in range(len(list_channels)-1):
            sequence.append(nn.Conv3d(in_channels=list_channels[k], out_channels=list_channels[k+1],
                                           kernel_size=(ks,ks,ks),stride=(1,1,1),padding=p))
            sequence.append(nn.BatchNorm3d(list_channels[k+1]))
            sequence.append(nn.ReLU(inplace=True))

        sequence.append(nn.Conv3d(in_channels=list_channels[-1], out_channels=1,
                                  kernel_size=(ks, ks, ks), stride=(1, 1, 1), padding=p))

        self.sequenceCNN = nn.Sequential(*sequence)
        self.activation= nn.ReLU(inplace=True)

    def forward(self,x):
        res = x
        y = self.sequenceCNN(x)
        y = y + res
        return self.activation(y)

def get_psf(kernel_size, sigma0, alpha, nview, ny,sy, sid):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.


    # # Calculate the 2-dimensional gaussian kernel which is
    # # the product of two gaussian distributions for two different
    # # variables (in this case called x and y)
    # gaussian_kernel = (1. / (2. * math.pi * variance)) * \
    #                   torch.exp(
    #                       -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
    #                       (2 * variance)
    #                   )
    # # Make sure sum of values in gaussian kernel equals 1.
    # gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    # psf = gaussian_kernel.repeat(ny, nview, 1, 1).permute(2, 3, 0, 1)

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


    psf_RM = get_psf(kernel_size=7,sigma0=1.1684338873367237,alpha=0.03235363042582603,nview=120,
                  ny=ny,sy=dy,sid = 280).to(device)

    # psf_noRM = get_psf(kernel_size=1,sigma0=0,alpha=0,nview=120,
    #               ny=ny,sy=dy,sid=280).to(device)

    A_RM = SPECT(size_in=(nx, ny, nz), size_out=(256, 256, nprojs),
              mumap=attmap_tensor, psfs=psf_RM, dy=dy)
    # A_noRM = SPECT(size_in=(nx, ny, nz), size_out=(256, 256, nprojs),
    #           mumap=attmap_tensor, psfs=psf_noRM, dy=dy)
    #

    # input = itk.imread(args.input)
    # input_tensor = torch.from_numpy(itk.array_from_image(input).astype(np.float32)).to(device)
    # input_tensor = input_tensor.permute(2,0,1)
    # input_tensor_fp = A._apply(input_tensor)
    # input_tensor_fp_rtk = projs_mir_to_rtk(input_tensor_fp.detach().cpu().numpy())
    # itk.imwrite(itk.image_from_array(input_tensor_fp_rtk), args.output)

    # MLEM reconstruction after 20 iterations
    print(f"p shape : {projs_tensor_mir.shape}")

    projs_tensor_mir = projs_tensor_mir.to(device)

    unet = CNN(nc=args.nc,ks=args.ks,nl=args.nl).to(device=device)
    print(unet)
    nb_params = sum(p.numel() for p in unet.parameters())
    print(f'NUMBER OF PARAMERS : {nb_params}')
    # loss,optimizer
    optimizer = optim.Adam(unet.parameters(), lr=args.lr)

    if args.loss == "L1":
        loss = torch.nn.L1Loss()
    elif args.loss=="KL":
        loss = torch.nn.KLDivLoss()
    elif args.loss=="PNLL":
        loss = torch.nn.PoissonNLLLoss(log_input=True)
    else:
        print(f"ERROR: unrecognized loss ({args.loss})")
        exit(0)

    print(projs_tensor_mir.dtype)
    # xn = mlem(x=x0,p=projs_tensor_mir,SPECT_sys=A,niter=args.niter, net = unet)
    # xn = deep_mlem_v3(p = projs_tensor_mir,
    #                SPECT_sys_RM=A_RM, SPECT_sys_noRM=A_noRM,
    #                niter=args.niter,net=unet,
    #                loss = loss,optimizer=optimizer)

    # xn = deep_mlem_v4(p=projs_tensor_mir,SPECT_sys_RM=A_RM,niter=args.niter,net=unet,loss=loss,optimizer=optimizer)
    xn = deep_mlem_v5(p=projs_tensor_mir,SPECT_sys_RM=A_RM,niter=args.niter,net=unet,loss=loss,optimizer=optimizer)

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
    parser.add_argument("--loss", type =str)
    parser.add_argument("--nc", type =int, default=8)
    parser.add_argument("--ks", type =int, default=3)
    parser.add_argument("--nl", type =int, default=6)
    parser.add_argument("--output")
    parser.add_argument("--iter")
    args = parser.parse_args()

    main()
