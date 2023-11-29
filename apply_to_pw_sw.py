#!/usr/bin/env python3

import argparse
import torch
import numpy as np
import itk

from apply import apply_to_input

from DeepPVC import Model_instance, helpers, helpers_params

def main():
    print(args)

    print(f'Apply the pth {args.pth} to the set of projections contained in : ')
    print(f'PW: {args.pw} \nPW_rec_fp {args.pw_rec_fp}\nSW: {args.sw} \nSW_rec_fp {args.sw_rec_fp}')

    device = helpers.get_auto_device(device_mode="cpu")
    pth_file = torch.load(args.pth, map_location=device)
    params = pth_file['params']
    helpers_params.check_params(params)

    params['jean_zay']=False

    model = Model_instance.ModelInstance(params=params, from_pth=args.pth,resume_training=False,device=device)
    model.load_model(pth_path=args.pth)
    model.switch_device(device)
    model.switch_eval()
    model.show_infos()

    output_pw = apply_to_input(input=args.pw,input_rec_fp=args.pw_rec_fp,attmap_fp=args.attmap_fp,params=params,device=device,model=model)
    output_sw = apply_to_input(input=args.sw,input_rec_fp=args.sw_rec_fp,attmap_fp=args.attmap_fp,params=params,device=device,model=model)


    # SCATTER CORRECTION DEW
    factor = 1.1
    array_projections_scatter_corrected = output_pw - factor * output_sw
    array_projections_scatter_corrected[array_projections_scatter_corrected<0]=0



    if args.pw[-3:] in ["mhd", "mha"]:
        input_image = itk.imread(args.pw)
        vSpacing = np.array(input_image.GetSpacing())
        vOffset = np.array(input_image.GetOrigin())

    output_image = itk.image_from_array(array_projections_scatter_corrected)
    output_image.SetSpacing(vSpacing)
    output_image.SetOrigin(vOffset)
    itk.imwrite(output_image, args.output)
    print(f'Done! output at : {args.output}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth")
    parser.add_argument("--pw")
    parser.add_argument("--pw_rec_fp")
    parser.add_argument("--sw")
    parser.add_argument("--sw_rec_fp")
    parser.add_argument("--attmap_fp")
    parser.add_argument("--output")
    args = parser.parse_args()
    main()
