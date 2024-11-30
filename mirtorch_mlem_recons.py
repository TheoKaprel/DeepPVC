#!/usr/bin/env python3

import argparse
import sys
import os

import matplotlib.pyplot as plt

host = os.uname()[1]

import torch

if (host !='suillus'):
    sys.path.append("/linkhome/rech/gencre01/uyo34ub/homeMocamed/WORK/PVE/MIRTorch")
else:
    sys.path.append("/export/home/tkaprelian/Desktop/External_repositories/MIRTorch")
from mirtorch.linear.spect import SPECT


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

def osem(x, p, psf_RM, img_size, nprojs,attmap,dy, nprojpersubset, niter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nx, ny, nz = img_size[0], img_size[1], img_size[2]
    list_A = []
    nsubsets = int(nprojs/nprojpersubset)
    for k in range(nsubsets):
        id = np.array([k + nsubsets * j for j in range(nprojpersubset)])
        psf_RM_k = psf_RM[:,:,:,id]
        print(psf_RM_k.shape)
        list_A.append(SPECT(size_in=(nx, ny, nz), size_out=(128, 128, nprojpersubset),
              mumap=attmap, psfs=psf_RM_k, dy=dy,first_angle = id[0]*360/nprojs))

    ones = torch.ones((128,128,nprojpersubset)).to(device)
    out = torch.clone(x)
    for iter in range(niter):
        for subs in range(nsubsets):
            SPECT_sys = list_A[subs]
            asum = SPECT_sys._apply_adjoint(ones)
            asum[asum == 0] = float('Inf')
            ybar = SPECT_sys._apply(out)
            id = torch.tensor([int(subs + nsubsets * j) for j in range(nprojpersubset)])
            yratio = torch.div(p[:,:,id], ybar)
            back = SPECT_sys._apply_adjoint(yratio)
            out = torch.multiply(out, torch.div(back, asum))
        print(f'iter : {iter}')
        itk.imwrite(itk.image_from_array(np.transpose(out.detach().cpu().numpy(),(1,2,0))), os.path.join(args.iter, f"iter_{iter}.mhd"))

    return out


def deep_mlem_v1(p, SPECT_sys_noRM, SPECT_sys_RM, niter, net, loss, optimizer):
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


def deep_mlem_v5(p, SPECT_sys_RM, niter, net, loss, optimizer, input):
    print('OSEM-RM')
    x_RM = input
    x_RM_max = x_RM.max()
    x_RM_n = x_RM/x_RM_max

    print("Training")
    for iter in range(niter):
        out_hat = net(x_RM_n[None, None, :, :, :])[0, 0, :, :, :]
        out_hat = out_hat * x_RM_max
        ybar = SPECT_sys_RM._apply(out_hat)
        loss_k = loss(ybar, p)
        print(f"loss {iter} : {loss_k}")
        loss_k.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        itk.imwrite(itk.image_from_array((out_hat.detach().cpu().numpy())), os.path.join(args.iter, f"iter_{iter}.mhd"))

    return out_hat

def deep_mlem_v6(p, SPECT_sys_noRM, SPECT_sys_RM, niter,nosem, net1,net2, loss, optimizer):
    if loss.__class__==torch.nn.L1Loss:
        p_max = p.max()
        p = p / p_max
        norm = "max"
    elif ((loss.__class__==torch.nn.KLDivLoss) or (loss.__class__==torch.nn.PoissonNLLLoss)):
        norm = "log"

    print(f"NORM : {norm}")

    for k in range(niter):
        p_hatn = net1(p[None,None,:,:,:])[0,0,:,:,:]
        p_hat = p_hatn * p_max if (norm=="max") else p_hatn
        del p_hatn
        asum = SPECT_sys_noRM._apply_adjoint(torch.ones_like(p))
        asum[asum == 0] = float('Inf')
        rec_corrected = torch.ones_like(asum)
        for _ in range(nosem):
            print(f"    inner {_}")
            ybar = SPECT_sys_noRM._apply(rec_corrected)
            ybar[ybar == 0] = float('Inf')
            yratio = torch.div(p_hat, ybar)
            del ybar
            back = SPECT_sys_noRM._apply_adjoint(yratio)
            del yratio
            rec_corrected = torch.multiply(rec_corrected, torch.div(back, asum))
            del back

        del asum

        rec_corrected_corrected = net2(rec_corrected[None,None,:,:,:])[0,0,:,:,:]


        rec_corrected_fp = SPECT_sys_RM._apply(rec_corrected_corrected)
        rec_corrected_fpn = (rec_corrected_fp / p_max) if (norm=="max") else torch.log(rec_corrected_fp+1e-8)
        loss_k = loss(rec_corrected_fpn, p)
        loss_k.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"loss {k} : {loss_k}")
        del rec_corrected,rec_corrected_fp
        itk.imwrite(itk.image_from_array(rec_corrected_corrected.detach().cpu().numpy()), os.path.join(args.iter, f"iter_{k}.mhd"))

    return rec_corrected_corrected

def deep_mlem_v7(p, net, loss, optimizer, input, psf_RM, img_size, nprojs,attmap,dy, nprojpersubset, niter):
    print('OSEM-RM')
    x_RM = input

    nx, ny, nz = img_size[0], img_size[1], img_size[2]
    list_A = []
    nsubsets = int(nprojs/nprojpersubset)
    for k in range(nsubsets):
        id = np.array([k + nsubsets * j for j in range(nprojpersubset)])
        psf_RM_k = psf_RM[:,:,:,id]
        print(psf_RM_k.shape)
        list_A.append(SPECT(size_in=(nx, ny, nz), size_out=(128, 128, nprojpersubset),
              mumap=attmap, psfs=psf_RM_k, dy=dy,first_angle = id[0]*360/nprojs))

    for iter in range(niter):
        loss_iter = 0
        for subs in range(nsubsets):
            SPECT_sys = list_A[subs]
            out_hat = net(x_RM[None, None, :, :, :])[0, 0, :, :, :]
            ybar = SPECT_sys._apply(out_hat)

            id = torch.tensor([int(subs + nsubsets * j) for j in range(nprojpersubset)])
            p_subs = p[:,:,id]

            loss_k = loss(ybar, p_subs)

            loss_iter+=loss_k/nsubsets
            loss_k.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        print(f"loss {iter} : {loss_iter}")

        if iter%args.saveevery==0:
            itk.imwrite(itk.image_from_array((out_hat.detach().cpu().numpy())),
                    os.path.join(args.iter, f"iter_{iter}.mhd"))
            if args.matrix:
                itk.imwrite(itk.image_from_array((net.M.detach().cpu().numpy())),
                        os.path.join(args.iter, f"matrix_{iter}.mhd"))

    return out_hat

def deep_mlem_v8(p, SPECT_sys_RM, SPECT_sys_noRM, niter, net, loss, optimizer):
    print("Training")
    bp_ones = SPECT_sys_noRM._apply_adjoint(torch.ones_like(p))
    bp_ones[bp_ones == 0] = float('Inf')
    for iter in range(niter):
        p_hat = net(p[None,None,:,:,:])[0,0,:,:,:]
        out_hat = SPECT_sys_noRM._apply_adjoint(p_hat) / bp_ones
        p_RM  = SPECT_sys_RM._apply(out_hat)
        loss_k = loss(p_RM, p)
        print(f"loss {iter} : {loss_k}")
        loss_k.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        itk.imwrite(itk.image_from_array((out_hat.detach().cpu().numpy())), os.path.join(args.iter, f"iter_{iter}.mhd"))
    return out_hat


def deep_mlem_v9(p, SPECT_sys_RM, SPECT_sys_noRM, niter, net1,net2, loss, optimizer):
    print("Training")
    bp_ones = SPECT_sys_noRM._apply_adjoint(torch.ones_like(p))
    bp_ones[bp_ones == 0] = float('Inf')
    for iter in range(niter):
        p_hat = net1(p[None,None,:,:,:])[0,0,:,:,:]
        out_hat_rec = SPECT_sys_noRM._apply_adjoint(p_hat) / bp_ones
        out_hat = net2(out_hat_rec[None,None,:,:,:])[0,0,:,:,:]
        p_RM  = SPECT_sys_RM._apply(out_hat)
        loss_k = loss(p_RM, p)
        print(f"loss {iter} : {loss_k}")
        loss_k.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        itk.imwrite(itk.image_from_array((out_hat.detach().cpu().numpy())), os.path.join(args.iter, f"iter_{iter}.mhd"))
    return out_hat


class HammingFilter:
    r"""Implementation of the Hamming filter given by :math:`\Pi(\omega) = \frac{1}{2}\left(1+\cos\left(\frac{\pi(|\omega|-\omega_L)}{\omega_H-\omega_L} \right)\right)` for :math:`\omega_L \leq |\omega| < \omega_H` and :math:`\Pi(\omega) = 1` for :math:`|\omega| \leq \omega_L` and :math:`\Pi(\omega) = 0` for :math:`|\omega|>\omega_H`. Arguments ``wl`` and ``wh`` should be expressed as fractions of the Nyquist frequency (i.e. ``wh=0.93`` represents 93% the Nyquist frequency).
    """

    def __init__(self, wl, wh):
        self.wl = wl / 2  # units of Nyquist Frequency
        self.wh = wh / 2

    def __call__(self, w):
        w = w.cpu().numpy()
        filter = np.piecewise(
            w,
            [np.abs(w) <= self.wl, (self.wl < np.abs(w)) * (self.wh >= np.abs(w)), np.abs(w) > self.wh],
            [lambda w: 1, lambda w: 1 / 2 * (1 + np.cos(np.pi * (np.abs(w) - self.wl) / (self.wh - self.wl))),
             lambda w: 0])
        return torch.tensor(filter)

class RampFilter:
    r"""Implementation of the Ramp filter :math:`\Pi(\omega) = |\omega|`
    """
    def __init__(self):
        return
    def __call__(self, w):
        return torch.abs(w)

import matplotlib.pyplot as plt

def fbp(p,SPECT_sys_noRM):
    freq_fft = torch.fft.fftfreq(p.shape[-2])
    filtr1 = HammingFilter(wl=0.5, wh=1)
    filtr2 = RampFilter()
    filter_total = filtr1(freq_fft)*filtr2(freq_fft)
    print(filter_total.shape)
    proj_filtered = torch.zeros_like(p)
    for i in range(128):
        for angle in range(120):
            proj_fft = torch.fft.fft(p[:,i,angle])
            proj_fft = proj_fft * filter_total
            proj_filtered[:,i,angle] = torch.fft.ifft(proj_fft).real

    fig,ax = plt.subplots(1,4)
    ax[0].imshow(p[:,:,0].detach().cpu().numpy())
    ax[1].imshow(proj_filtered[:,:,0].detach().cpu().numpy())
    ax[2].imshow(proj_filtered[:,:,64].detach().cpu().numpy())
    ax[3].plot(filter_total.detach().cpu().numpy())
    plt.show()

    bp_ones = SPECT_sys_noRM._apply_adjoint(torch.ones_like(proj_filtered))
    bp_ones[bp_ones == 0] = float('Inf')

    a = torch.nn.ReLU()

    return a(SPECT_sys_noRM._apply_adjoint(proj_filtered) / bp_ones)


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
        y = self.sequenceCNN(x.clone())
        y = y + res
        return self.activation(y)


class Mult(nn.Module):
    def __init__(self, input_size):
        super(Mult, self).__init__()
        self.M = torch.nn.Parameter(torch.ones(input_size,requires_grad=True))
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x * self.M)


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


    A_RM = SPECT(size_in=(nx, ny, nz), size_out=(128, 128, nprojs),
              mumap=attmap_tensor, psfs=psf_RM, dy=dy,first_angle=0)
    print(f"PSF shape: {psf_RM.shape}")

    # input = itk.imread(args.input)
    # input_tensor = torch.from_numpy(itk.array_from_image(input).astype(np.float32)).to(device)
    # input_tensor = input_tensor.permute(2,0,1)
    # input_tensor_fp = A._apply(input_tensor)
    # input_tensor_fp_rtk = projs_mir_to_rtk(input_tensor_fp.detach().cpu().numpy())
    # itk.imwrite(itk.image_from_array(input_tensor_fp_rtk), args.output)

    # MLEM reconstruction after 20 iterations
    print(f"p shape : {projs_tensor_mir.shape}")

    projs_tensor_mir = projs_tensor_mir.to(device)


    if args.loss == "L1":
        loss = torch.nn.L1Loss()
    elif args.loss=="L2":
        loss = torch.nn.MSELoss()
    elif args.loss=="KL":
        loss = torch.nn.KLDivLoss()
    elif args.loss=="PNLL":
        loss = torch.nn.PoissonNLLLoss(log_input=False,eps=1e-4)
        # loss = lambda inp,targ: (inp - targ * torch.log(inp+1e-8)).mean()

    else:
        print(f"ERROR: unrecognized loss ({args.loss})")
        exit(0)



    if args.version in [6,9]:
        unet1 = CNN(nc=args.nc, ks=args.ks, nl=args.nl).to(device=device)
        unet2 = CNN(nc=args.nc, ks=args.ks, nl=args.nl).to(device=device)
        print("------------ unet 1 -----------------")
        print(unet1)
        nb_params = sum(p.numel() for p in unet1.parameters())
        print(f'NUMBER OF PARAMERS : {nb_params}')
        print("------------ unet 2 -----------------")
        print(unet2)
        nb_params = sum(p.numel() for p in unet2.parameters())
        print(f'NUMBER OF PARAMERS : {nb_params}')

        # loss,optimizer
        optimizer = optim.Adam(list(unet1.parameters())+list(unet2.parameters()), lr=args.lr)
    elif args.version>0:
        if args.matrix:
            unet = Mult(input_size=(128, 128, 128)).to(device=device)
        else:
            unet = CNN(nc=args.nc, ks=args.ks, nl=args.nl).to(device=device)

        # unet = torch.compile(unet)
        print(unet)
        nb_params = sum(p.numel() for p in unet.parameters())
        print(f'NUMBER OF PARAMERS : {nb_params}')
        # loss,optimizer
        optimizer = optim.Adam(unet.parameters(), lr=args.lr)

    if args.version in [1,2,3,6,8,9,10]:
        psf_noRM = get_psf(kernel_size=1,sigma0=0,alpha=0,nview=120,
                      ny=ny,sy=dy,sid=280).to(device)

        A_noRM = SPECT(size_in=(nx, ny, nz), size_out=(128, 128, nprojs),
                  mumap=attmap_tensor, psfs=psf_noRM, dy=dy,first_angle = 0)


    print(projs_tensor_mir.dtype)
    # xn = mlem(x=x0,p=projs_tensor_mir,SPECT_sys=A,niter=args.niter, net = unet)
    # xn = deep_mlem_v3(p = projs_tensor_mir,
    #                SPECT_sys_RM=A_RM, SPECT_sys_noRM=A_noRM,
    #                niter=args.niter,net=unet,
    #                loss = loss,optimizer=optimizer)

    # xn = deep_mlem_v4(p=projs_tensor_mir,SPECT_sys_RM=A_RM,niter=args.niter,net=unet,loss=loss,optimizer=optimizer)


    if args.version==0:
        x0 = torch.ones_like(A_RM.mumap)
        # xn = mlem(x=x0,p=projs_tensor_mir,SPECT_sys=A_RM,niter=args.niter)
        xn = osem(x0, p=projs_tensor_mir, psf_RM=psf_RM, img_size=(nx,ny,nz), nprojs=120, attmap=attmap_tensor, dy=dy, nprojpersubset=args.nprojpersubset, niter=args.niter)

    elif args.version==5:
        input = itk.imread(args.input)
        input = itk.array_from_image(input).astype(np.float32)
        input = np.transpose(input, (2, 0, 1))
        input = torch.from_numpy(input).to(device)
        xn = deep_mlem_v5(p=projs_tensor_mir,SPECT_sys_RM=A_RM,niter=args.niter,net=unet,loss=loss,optimizer=optimizer, input = input)
    elif args.version==7:
        input = itk.imread(args.input)
        input = itk.array_from_image(input).astype(np.float32)
        input = np.transpose(input, (2, 0, 1))
        input = torch.from_numpy(input).to(device)
        xn =  deep_mlem_v7(p=projs_tensor_mir, net=unet, loss=loss, optimizer=optimizer, input=input, psf_RM=psf_RM, img_size=(nx,ny,nz), nprojs=120, attmap=attmap_tensor, dy=dy, nprojpersubset=args.nprojpersubset, niter=args.niter)
    elif args.version == 6:
        xn = deep_mlem_v6(p=projs_tensor_mir,SPECT_sys_RM=A_RM,SPECT_sys_noRM=A_noRM,niter=args.niter,nosem=args.nosem,
                     net1=unet1,net2=unet2,loss=loss,optimizer=optimizer)
    elif args.version==8:
        xn = deep_mlem_v8(p=projs_tensor_mir, SPECT_sys_RM=A_RM, SPECT_sys_noRM=A_noRM, niter=args.niter, net=unet, loss=loss,
                     optimizer=optimizer)
    elif args.version==9:
        xn = deep_mlem_v9(p=projs_tensor_mir,SPECT_sys_RM=A_RM, SPECT_sys_noRM=A_noRM,niter=args.niter,
                          net1 = unet1, net2 = unet2, loss = loss, optimizer=optimizer)
    elif args.version==10:
        xn = fbp(p = projs_tensor_mir,SPECT_sys_noRM=A_noRM)

    rec_array = xn.detach().cpu().numpy()
    rec_array_ = np.transpose(rec_array, (1,2,0))
    rec_itk  =itk.image_from_array(rec_array_)
    itk.imwrite(rec_itk, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
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
    parser.add_argument("--nosem" ,type = int)
    parser.add_argument("--saveevery" ,type = int)
    parser.add_argument("--nprojpersubset" ,type = int)
    parser.add_argument("--version", type = int)
    parser.add_argument("--matrix",action="store_true")
    args = parser.parse_args()

    main()
