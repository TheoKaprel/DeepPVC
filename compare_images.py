import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import itk
import click
import glob
import os

def get_ref(file):
    i1 = file.rfind('DeepPVC_') + 8
    i2 = i1
    while file[i2]!='.':
        i2+=1
    return file[i1:i2]

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--folder')
@click.option('--auto', is_flag = True, default = False, help = 'If --auto, the selected images will be {ref}.mhd for the source and {ref}_rec_PVE_PVC.mhd, {ref}_rec_PVE_noPVC.mhd, {ref}_rec_noPVE_noPVC.mhd and all the {ref}_rec_PVE_DeepPVC_*.mhd')
@click.option('--ref')
@click.option('--type', default = 'mhd', help = "mhd or mha", show_default = True)
@click.option('--source', help = 'If not --auto mode, specify the source here')
@click.option('--image', '-i', multiple = True, help = 'If not --auto mode, specify images to compare')
@click.option('--slice', type = int, multiple = True)
@click.option('--profile', type = int, multiple = True)
@click.option('--error', is_flag = True, default = False)
def comparison_click(folder, auto, ref,type,source, image, slice, profile, error):
    if auto:
        comparison_auto(folder, ref,type,slice, profile)
    else:
        comparison_manual(folder, source, image, slice, profile,error)

def comparison_auto(folder, ref,type,slice, profile):

    src_file = os.path.join(folder, f'{ref}.{type}')
    img_rec_PVE_PVC_file = os.path.join(folder, f'{ref}_rec_PVE_PVC.{type}')
    img_rec_PVE_noPVC_file = os.path.join(folder, f'{ref}_rec_PVE_noPVC.{type}')
    img_rec_noPVE_noPVC_file = os.path.join(folder, f'{ref}_rec_noPVE_noPVC.{type}')
    list_of_img_rec_DeepPVC_file = glob.glob( os.path.join(folder, f'{ref}_rec_PVE_DeepPVC_*.{type}'))
    nDeepPVC = len(list_of_img_rec_DeepPVC_file)
    list_refs_pix2pix = [get_ref(imgdeepfile) for imgdeepfile in list_of_img_rec_DeepPVC_file]

    img_src = itk.array_from_image(itk.imread(src_file))
    img_rec_PVE_PVC = itk.array_from_image(itk.imread(img_rec_PVE_PVC_file))
    img_rec_PVE_noPVC = itk.array_from_image(itk.imread(img_rec_PVE_noPVC_file))
    img_rec_noPVE_noPVC = itk.array_from_image(itk.imread(img_rec_noPVE_noPVC_file))

    list_of_img_rec_DeepPVC = []
    for i in range(nDeepPVC):
        list_of_img_rec_DeepPVC.append(itk.array_from_image(itk.imread(list_of_img_rec_DeepPVC_file[i])))

    # MSE
    error_noPVC = np.mean((img_src - img_rec_PVE_noPVC) ** 2)
    error_PVC = np.mean((img_src - img_rec_PVE_PVC) ** 2)
    error_noPVE_noPVC = np.mean((img_src - img_rec_noPVE_noPVC) ** 2)
    list_error_DeepPVC = [np.mean((img_src - imgDeep) ** 2) for imgDeep in list_of_img_rec_DeepPVC]



    list_of_all_images = [img_src, img_rec_PVE_PVC, img_rec_PVE_noPVC, img_rec_noPVE_noPVC]
    list_of_all_images+=list_of_img_rec_DeepPVC

    fig,ax = plt.subplots()
    imgs = ['noPVC', 'PVC']
    imgs+=list_refs_pix2pix
    imgs.append('noPVE_noPVC')
    errs = [error_noPVC, error_PVC]
    errs+=list_error_DeepPVC
    errs.append(error_noPVE_noPVC)
    ax.bar(imgs, errs)
    ax.set_ylabel('MSE')



    if (slice==None or len(slice)==0):
        idxmax = np.argwhere(img_rec_PVE_PVC==np.max(img_rec_PVE_PVC))
        slice,_,__ = idxmax[0]
        slice = (slice,)



    for s in slice:
        vmax = max([np.max(img[s, :, :]) for img in list_of_all_images])
        vmin = min([np.min(img[s, :, :]) for img in list_of_all_images])

        fig,ax = plt.subplots(2,nDeepPVC+2)
        ax[0,0].imshow(img_src[s,:,:], vmin = vmin, vmax=vmax)
        ax[0,0].set_title('Source')
        ax[0,1].imshow(img_rec_PVE_noPVC[s,:,:], vmin = vmin, vmax=vmax)
        ax[0,1].set_title('PVE/noPVC')
        ax[0,2].imshow(img_rec_PVE_PVC[s,:,:], vmin = vmin, vmax=vmax)
        ax[0,2].set_title('PVE/PVC')
        ax[1,0].imshow(img_rec_noPVE_noPVC[s,:,:], vmin = vmin, vmax=vmax)
        ax[1,0].set_title('noPVE/noPVC')

        for i in range(nDeepPVC):
            ax[1,i+1].imshow(list_of_img_rec_DeepPVC[i][s,:,:], vmin = vmin, vmax=vmax)
            ref_i = list_refs_pix2pix[i]
            ax[1,i+1].set_title(f'DeepPVC_{ref_i}')

        plt.suptitle(f'Slice : {s}')

        if len(profile)>0:
            nb_profiles = len(profile)
            fig_pr,ax_pr = plt.subplots(nb_profiles,1)
            if nb_profiles==1:
                ax_pr = [ax_pr]
            for p in range(nb_profiles):
                ax_pr[p].plot(img_src[s,profile[p],:], label = 'Source')
                ax_pr[p].plot(img_rec_PVE_noPVC[s,profile[p],:], label = 'PVE/noPVC')
                ax_pr[p].plot(img_rec_PVE_PVC[s,profile[p],:], label = 'PVE/PVC')
                ax_pr[p].plot(img_rec_noPVE_noPVC[s,profile[p],:], label = 'noPVE/noPVC')
                for i in range(nDeepPVC):
                    ax_pr[p].plot(list_of_img_rec_DeepPVC[i][s,profile[p],:], label = f'DeepPVC_{list_refs_pix2pix[i]}')
    plt.legend()
    plt.show()


def comparison_manual(folder, source, image, slice,profile, error):
    src_fn = os.path.join(folder, source)
    img_src = itk.array_from_image(itk.imread(src_fn))
    norm = np.sum(img_src**2)
    list_of_all_images = []
    list_of_labels = []
    for i in image:
        img_fn = os.path.join(folder, i)
        list_of_all_images.append(itk.array_from_image(itk.imread(img_fn)))
        list_of_labels.append(i)

    list_of_mse = []
    for img in list_of_all_images:
        list_of_mse.append(np.sum((img - img_src) **2) / norm)


    # list_of_labels = ['PVE/noPVC', 'PVE/PVC','noPVE/noPVC', 'PVE/DeepPVC', 'noPVE/noPVC (298 pixels detector)']

    fig_mse, ax_mse = plt.subplots()
    ax_mse.bar([k for k in range(len(list_of_all_images))],list_of_mse, tick_label = list_of_labels, color = 'black')
    ax_mse.set_ylabel('MSE', fontsize = 20)
    # plt.xticks(fontsize=20, rotation=0)





    if (slice==None or len(slice)==0):
        idxmax = np.argwhere(list_of_all_images[0]==np.max(list_of_all_images[0]))
        slice,_,__ = idxmax[0]
        slice = (slice,)



    for s in slice:
        vmax = max([np.max(img[s, :, :]) for img in list_of_all_images])
        # vmin = min([np.min(img[s, :, :]) for img in list_of_all_images])
        vmin = -vmax/8

        plt.figure(figsize=(16, 11))
        plt.subplots_adjust(left=0.01,
                            bottom=0.03,
                            right=0.852,
                            top=0.965,
                            wspace=0,
                            hspace=0.107)

        ncols = 3
        nrows = (len(list_of_all_images)+1) // ncols + ((len(list_of_all_images)+1) % ncols > 0)
        ax = plt.subplot(nrows,ncols,1)
        ax.imshow(img_src[s,:,:], vmin = vmin, vmax = vmax, cmap = plt.get_cmap('inferno'))
        ax.set_title('Source', fontsize = 20)
        ax.set_xlabel("")


        for n, img in enumerate(list_of_all_images):
            ax = plt.subplot(nrows, ncols, n + 2)
            im = ax.imshow(img[s,:,:],vmin = vmin, vmax=vmax, cmap=plt.get_cmap('inferno'))
            ax.set_title(list_of_labels[n], fontsize = 20)
            ax.set_xlabel("")


        cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        plt.colorbar(mappable=im, cax=cax)


        for pr in profile:
            fig_pr,ax_pr = plt.subplots()
            ax_pr.plot(img_src[s,pr,:], label = 'src')
            for n,img in enumerate(list_of_all_images):
                ax_pr.plot(img[s,pr,:], label = list_of_labels[n])
            plt.legend()





        if error:
            plt.figure(figsize=(15, 12))
            plt.subplots_adjust(hspace=0.2)
            plt.suptitle('ERROR images')

            ncols = 3
            nrows = len(list_of_all_images) // ncols + (len(list_of_all_images) % ncols > 0)
            emax = max([np.max((img[s, :, :] - img_src[s,:,:])**2) for img in list_of_all_images])

            for n, img in enumerate(list_of_all_images):
                error_img = (img - img_src) **2
                ax = plt.subplot(nrows, ncols, n + 1)

                ax.imshow(error_img[s, :, :], vmin=0, vmax=emax)
                ax.set_title(list_of_labels[n])
                ax.set_xlabel("")


        plt.show()


if __name__=='__main__':
    comparison_click()