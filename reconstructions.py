import click
import os
import json
import itk
from itk import RTK as rtk

import apply

import sys
sys.path.append("/export/home/tkaprelian/Desktop/PVE/PVE_data")
from Analytical_data import parameters

def get_ref(pth, ffrom='filename'):
    if ffrom=='filename':
        i1 = pth.rfind('pix2pix_') + 8
        i2 = i1
        while pth[i2]!='_':
            i2+=1
        return pth[i1:i2]
    elif ffrom=='json':
        json_filename = pth[:-4]+'.json'
        with open(json_filename, "r") as f:
            json_data = json.load(f)
            ref = json_data['ref']
            return ref
    else:
        print(f'ERROR : Wrong way to get the ref of {pth}. Specify ffrom = ...')
        exit(0)



CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--folder')
@click.option('--like')
@click.option('--nprojpersubset', type = int, default = 10)
@click.option('--niterations', type = int, default = 5)
@click.option('--ref', help = 'In the folder there should be ref.mhd, ref_PVE.mhd, ref_PVfree.mhd')
@click.option('--type', default = "mhd", show_default = True, help = "mha or mhd")
@click.option('--pth', help = 'model to apply to the PVE projections before reconstruction (if --deeppvc') # 'path/to/saved/model.pth'
@click.option('--nopvc',is_flag = True, default = False, help = 'Reconstruct the image from the PVE sinogram without any PVC algorithm')
@click.option('--pvc',is_flag = True, default = False, help = 'Reconstruct the image from the PVE sinogram with a classical PVC algorithm (Zeng)')
@click.option('--deeppvc',is_flag = True, default = False, help = 'Reconstruct the image from Pix2Pix corrected projections')
@click.option('--nopve_nopvc',is_flag = True, default = False, help = 'Reconstruct the image from PVfree projections without correction during reconstruction')
@click.option('--data_folder', help = 'Location of the folder containing : geom_120.xml and acf_ct_air.mhd')
def reconstructions_click(pth,folder,like,nprojpersubset,niterations, ref, type,nopvc, pvc, deeppvc,nopve_nopvc, data_folder):
    reconstructions(pth, folder,like,nprojpersubset,niterations, ref,type,nopvc, pvc, deeppvc,nopve_nopvc, data_folder)

def reconstructions(pth, folder,like,nprojpersubset,niterations, ref,type,nopvc, pvc, deeppvc,nopve_nopvc, data_folder):
    #Types
    Dimension = 3
    pixelType = itk.F
    imageType = itk.Image[pixelType, Dimension]

    # Volume
    like_image = itk.imread(like, pixelType)
    volume_image = rtk.ConstantImageSource[imageType].New()
    volume_image.SetSpacing(like_image.GetSpacing())
    volume_image.SetOrigin(like_image.GetOrigin())
    volume_image.SetSize(itk.size(like_image))
    volume_image.SetConstant(1)

    # Input Projections
    if (pvc or nopvc or deeppvc):
        proj_PVE_path = os.path.join(folder, f'{ref}_PVE.{type}')
        projections_PVE = itk.imread(proj_PVE_path, pixelType)
        nproj = itk.size(projections_PVE)[2]
    if nopve_nopvc:
        proj_PVfree_path = os.path.join(folder, f'{ref}_PVfree.{type}')
        projections_PVfree = itk.imread(proj_PVfree_path, pixelType)
        nproj = itk.size(projections_PVfree)[2]

    # Geometry
    geom_filename = os.path.join(data_folder, f'geom_{nproj}.xml')
    xmlReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    xmlReader.SetFilename(geom_filename)
    xmlReader.GenerateOutputInformation()
    geometry = xmlReader.GetOutputObject()

    # Attenuation
    attmap_filename = os.path.join(data_folder, f'acf_ct_air.mhd')
    attenuation_map = itk.imread(attmap_filename, pixelType)

    # Common OSEM parameters
    OSEMType = rtk.OSEMConeBeamReconstructionFilter[imageType, imageType]
    osem = OSEMType.New()
    osem.SetInput(0, volume_image.GetOutput())
    osem.SetInput(2, attenuation_map)
    osem.SetGeometry(geometry)
    osem.SetNumberOfIterations(niterations)
    osem.SetNumberOfProjectionsPerSubset(nprojpersubset)
    osem.SetForwardProjectionFilter(osem.ForwardProjectionType_FP_ZENG)
    osem.SetBackProjectionFilter(osem.BackProjectionType_BP_ZENG)
    osem.SetBetaRegularization(0)

    if pvc:
        print('Reconstruction with PVE and PVC...')
        output_PVE_PVC_filename = os.path.join(folder, f'{ref}_rec_PVE_PVC.{type}')

        osem.SetInput(1, projections_PVE)
        sigmaPVC = parameters.sigma0pve_default
        alphaPVC = parameters.alphapve_default
        print("with : ")
        print(f"sigma : {sigmaPVC}")
        print(f"alpha : {alphaPVC}")
        osem.SetSigmaZero(sigmaPVC)
        osem.SetAlpha(alphaPVC)
        osem.Update()
        itk.imwrite(osem.GetOutput(), output_PVE_PVC_filename)
        print(f'Done ! Output: {output_PVE_PVC_filename}')

    if nopvc:
        print('Reconstruction with PVE but without PVC...')
        output_PVE_noPVC_filename = os.path.join(folder, f'{ref}_rec_PVE_noPVC.{type}')

        osem.SetInput(1, projections_PVE)
        osem.SetSigmaZero(0)
        osem.SetAlpha(0)
        osem.Update()
        itk.imwrite(osem.GetOutput(), output_PVE_noPVC_filename)
        print(f'Done ! Output: {output_PVE_noPVC_filename}')

    if nopve_nopvc:
        print('Reconstruction without PVE and without PVC...')
        output_noPVE_noPVC_filename = os.path.join(folder, f'{ref}_rec_noPVE_noPVC.{type}')

        osem.SetInput(1, projections_PVfree)
        osem.SetSigmaZero(0)
        osem.SetAlpha(0)
        osem.Update()
        itk.imwrite(osem.GetOutput(), output_noPVE_noPVC_filename)
        print(f'Done ! Output: {output_noPVE_noPVC_filename}')

    if deeppvc:
        print('Reconstruction with DeepPVC corrected projections...')
        ref_pix2pix = get_ref(pth, ffrom='json')

        proj_DeepPVC_path = os.path.join(folder, f'{ref}_projDeepPVC_{ref_pix2pix}.{type}')

        apply.apply(pth=pth, input=proj_PVE_path, output_filename=proj_DeepPVC_path)

        projections_DeepPVC = itk.imread(proj_DeepPVC_path, pixelType)

        output_PVE_DeepPVC_filename = os.path.join(folder, f'{ref}_rec_PVE_DeepPVC_{ref_pix2pix}.{type}')

        osem.SetInput(1, projections_DeepPVC)
        osem.SetSigmaZero(0)
        osem.SetAlpha(0)
        osem.Update()
        itk.imwrite(osem.GetOutput(), output_PVE_DeepPVC_filename)

        print(f'Done ! Output: {output_PVE_DeepPVC_filename}')


if __name__ =='__main__':

    reconstructions_click()
