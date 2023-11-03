#!/usr/bin/env python3

import argparse
import torch
import numpy as np
import itk

from DeepPVC import Model_instance, helpers_data, helpers, helpers_params

def main():
    print(args)

    print(f'Apply the pth {args.pth} to the set of projections contained in : ')
    print(f'PW: {args.pw} \nPW_rec_fp {args.pw_rec_fp}\nSW: {args.sw} \nSW_rec_fp {args.sw_rec_fp}')

    device = helpers.get_auto_device(device_mode="cpu")
    pth_file = torch.load(args.pth, map_location=device)
    params = pth_file['params']
    helpers_params.check_params(params)

    data_normalisation = params['data_normalisation']
    params['jean_zay']=False

    model = Model_instance.ModelInstance(params=params, from_pth=args.pth,resume_training=False,device=device)
    model.load_model(pth_path=args.pth)
    model.switch_device(device)
    model.switch_eval()
    model.show_infos()

    output_pw = apply_to_input(input=args.pw,input_rec_fp=args.pw_rec_fp,params=params,device=device,model=model)
    output_sw = apply_to_input(input=args.sw,input_rec_fp=args.sw_rec_fp,params=params,device=device,model=model)


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

def apply_to_input(input, input_rec_fp, params, device, model):

    input_PVE_noisy_array = itk.array_from_image(itk.imread(input))
    input_rec_fp_array = itk.array_from_image(itk.imread(input_rec_fp)) if ((input_rec_fp is not None) and (params['with_rec_fp'])) else None

    with torch.no_grad():
        data_input = helpers_data.get_dataset_for_eval(params=params,
                                                       input_PVE_noisy_array=input_PVE_noisy_array,
                                                       input_rec_fp_array=input_rec_fp_array)


        print(f'input shape : {[data.shape for data in data_input]}')
        data_input = tuple([data.to(device) for data in data_input])


        data_normalisation = params['data_normalisation']
        norm_input = helpers_data.compute_norm_eval(dataset_or_img=data_input, data_normalisation=data_normalisation)
        if (data_normalisation!='none'):
            print(f'norm : {norm_input}')
        normed_input = helpers_data.normalize_eval(dataset_or_img=data_input, data_normalisation=data_normalisation,
                                                     norm=norm_input, params=model.params, to_torch=False)

        normed_output = model.forward(normed_input)
        denormed_output = helpers_data.denormalize_eval(dataset_or_img=normed_output,
                                                          data_normalisation=data_normalisation,
                                                          norm=norm_input, params=model.params, to_numpy=False)

        print(f'network output shape : {denormed_output.shape}')

        output_array = helpers_data.back_to_input_format(params=params,output=denormed_output)
        return output_array



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth")
    parser.add_argument("--pw")
    parser.add_argument("--pw_rec_fp")
    parser.add_argument("--sw")
    parser.add_argument("--sw_rec_fp")
    parser.add_argument("--output")
    args = parser.parse_args()
    main()
