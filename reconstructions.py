import click
import apply
import os
import subprocess
import json

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
@click.option('--ref', help = 'In the folder there should be ref.mhd, ref_PVE.mhd, ref_')
@click.option('--pth', help = 'model to apply to the PVE projections before reconstruction (if --deeppvc') # 'path/to/saved/model.pth'
@click.option('--nopvc',is_flag = True, default = False, help = 'Reconstruct the image from the PVE sinogram without any PVC algorithm')
@click.option('--pvc',is_flag = True, default = False, help = 'Reconstruct the image from the PVE sinogram with a classical PVC algorithm (Zeng)')
@click.option('--deeppvc',is_flag = True, default = False, help = 'Reconstruct the image from Pix2Pix corrected projections')
@click.option('--nopve_nopvc',is_flag = True, default = False, help = 'Reconstruct the image from PVfree projections without correction during reconstruction')
@click.option('--data_folder', help = 'Location of the folder containing : geom_60.xml and acf_ct_air.mhd')
def reconstructions_click(pth,folder,ref,nopvc, pvc, deeppvc,nopve_nopvc, data_folder):
    reconstructions(pth, folder, ref,nopvc, pvc, deeppvc,nopve_nopvc, data_folder)

def reconstructions(pth, folder, ref,nopvc, pvc, deeppvc,nopve_nopvc, data_folder):

    src_img_file =  f'{ref}.mhd'
    src_img_path = os.path.join(folder,src_img_file)
    proj_PVE_file = f'{ref}_PVE.mhd'
    proj_PVE_path = os.path.join(folder,proj_PVE_file)


    geom = os.path.join(data_folder, 'geom_120.xml')
    attmap = os.path.join(data_folder, 'acf_ct_air.mhd')

    if pvc:
        # Reconstruction with classical PVC
        output_PVE_PVC = os.path.join(folder, f'{ref}_rec_PVE_PVC.mhd')
        recPVE_PVC = subprocess.run(
            ['rtkosem',"-v", "-g", geom,"-o", output_PVE_PVC,"--path",folder,"--regexp", proj_PVE_file,
             "--like", src_img_path,"-f", "Zeng", "-b", "Zeng", "--attenuationmap", attmap,
             "--sigmazero", "0.9008418065898374", "--alphapsf", "0.025745123547513887"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if (recPVE_PVC.returncode != 0):
            print(f'ERROR in the reconstruction PVE/PVC')
            print(recPVE_PVC.stdout)
            print(recPVE_PVC.stderr)
            exit()
        else:
            print(f'Reconstruction done ! Output: {output_PVE_PVC}')

    if nopvc:
        # Reconstruction with no PVC
        output_PVE_noPVC = os.path.join(folder, f'{ref}_rec_PVE_noPVC.mhd')
        recPVE_noPVC = subprocess.run(
            ['rtkosem',"-v", "-g", geom,"-o", output_PVE_noPVC,"--path",folder,"--regexp", proj_PVE_file,
             "--like", src_img_path,"-f", "Zeng", "-b", "Zeng", "--attenuationmap", attmap,
             "--sigmazero", "0", "--alphapsf", "0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if (recPVE_noPVC.returncode != 0):
            print(f'ERROR in the reconstruction PVE/noPVC')
            print(recPVE_noPVC.stdout)
            print(recPVE_noPVC.stderr)
            exit()
        else:
            print(f'Reconstruction done ! Output: {output_PVE_noPVC}')


    if nopve_nopvc:
        # Reconstruction with PVfree projections
        proj_PVfree_file = f'{ref}_PVfree.mhd'
        output_noPVE_noPVC = os.path.join(folder, f'{ref}_rec_noPVE_noPVC.mhd')
        recPVE_noPVE_noPVC = subprocess.run(
            ['rtkosem',"-v", "-g", geom,"-o", output_noPVE_noPVC,"--path",folder,"--regexp", proj_PVfree_file,
             "--like", src_img_path,"-f", "Zeng", "-b", "Zeng", "--attenuationmap", attmap,
             "--sigmazero", "0", "--alphapsf", "0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if (recPVE_noPVE_noPVC.returncode != 0):
            print(f'ERROR in the reconstruction noPVE/noPVC')
            print(recPVE_noPVE_noPVC.stdout)
            print(recPVE_noPVE_noPVC.stderr)
            exit()
        else:
            print(f'Reconstruction done ! Output: {output_noPVE_noPVC}')


    if deeppvc:
        # Reconstruction with DeepPVC projections

        ref_pix2pix = get_ref(pth, ffrom='json')

        proj_DeepPVC_file = f'{ref}_projDeepPVC_{ref_pix2pix}.mhd'
        proj_DeepPVC_path = os.path.join(folder, proj_DeepPVC_file)

        apply.apply(pth=pth, input=proj_PVE_path, output_filename=proj_DeepPVC_path)

        output_PVE_DeepPVC = os.path.join(folder, f'{ref}_rec_PVE_DeepPVC_{ref_pix2pix}.mhd')
        recPVE_DeepPVC = subprocess.run(
            ['rtkosem',"-v", "-g", geom,"-o", output_PVE_DeepPVC,"--path",folder,"--regexp", proj_DeepPVC_file,
             "--like", src_img_path,"-f", "Zeng", "-b", "Zeng", "--attenuationmap", attmap,
             "--sigmazero", "0", "--alphapsf", "0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if (recPVE_DeepPVC.returncode != 0):
            print(f'ERROR in the reconstruction PVE/DeepPVC')
            print(recPVE_DeepPVC.stdout)
            print(recPVE_DeepPVC.stderr)
            exit()
        else:
            print(f'Reconstruction done ! Output: {output_PVE_DeepPVC}')



if __name__ =='__main__':

    reconstructions_click()
