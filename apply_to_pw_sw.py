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
    with_rec_fp = params['with_rec_fp']
    with torch.no_grad():
        if ('sino' in params and not params['full_sino']):
            # sino
            projs_input_ = itk.array_from_image(itk.imread(input)) #(120,256,256)
            nb_sino = params['sino']
            projs_input_t = projs_input_.transpose((1,0,2)) # (256,120,256)
            nb_projs_per_img = projs_input_t.shape[0]
            adjacent_channels_id = np.array([0]+[(-k) % nb_projs_per_img for k in range(1, nb_sino // 2 + 1)] +
                                            [(k) % nb_projs_per_img for k in range(1, nb_sino // 2 + 1)])

            projs_input = np.zeros((nb_projs_per_img,nb_sino+1, projs_input_t.shape[-2], projs_input_t.shape[-1]))
            for proj_i in range(nb_projs_per_img):
                proj_channels = (adjacent_channels_id+proj_i)%nb_projs_per_img
                projs_input[proj_i] = projs_input_t[proj_channels,:,:]
            # (256, 7,120,256)
            if with_rec_fp:
                projs_rec_fp = itk.array_from_image(itk.imread(input_rec_fp)).transpose((1,0,2))[:,None,:,:] # (256,1,120,256)
                projs_input = np.concatenate((projs_input,projs_rec_fp),axis=1)

            zeros_padding = np.zeros((projs_input.shape[0], projs_input.shape[1], 4, projs_input.shape[3]))
            projs_input = np.concatenate((zeros_padding,projs_input,zeros_padding), axis=2)

            projs_input = torch.Tensor(projs_input)
            # end sino

        elif params['full_sino']:
            projs_input = torch.Tensor(itk.array_from_image(itk.imread(input)).astype(np.float64))[None,:,:,:]
            if 'sino' in params:
                pad = torch.nn.ConstantPad2d((0, 0, 4, 4), 0)
                projs_input = pad(projs_input.transpose((0,2,1,3)))

            if with_rec_fp:
                img_rec_fp = torch.Tensor(itk.array_from_image(itk.imread(input_rec_fp))[None,:,:,:])
                if 'sino' in params:
                    img_rec_fp = pad(img_rec_fp.transpose((0, 2, 1, 3)))

                data_input = (projs_input,img_rec_fp)
            else:
                data_input = (projs_input)
        else:
            data_input = helpers_data.load_image(filename=input, is_ref=False, type=None, params=params)
            if with_rec_fp:
                img_rec_fp = torch.Tensor(
                    itk.array_from_image(itk.imread(input_rec_fp))[:, None, :, :])  # (120,1,256,256)
                data_input = torch.cat((data_input, img_rec_fp), dim=1)  # (120,7,256,256)


        print(f'input shape : {[data.shape for data in data_input]}')


        # norm_input = helpers_data.compute_norm_eval(dataset_or_img=projs_input, data_normalisation=data_normalisation)
        # if data_normalisation!='none':
        #     print(f'norm shape : {norm_input[0].shape}')
        # normed_input = helpers_data.normalize_eval(dataset_or_img=projs_input, data_normalisation=data_normalisation,
        #                                              norm=norm_input, params=model.params, to_torch=False)
        data_input = tuple([data.to(device) for data in data_input])
        denormed_output_i = model.forward(data_input)
        # denormed_output_i = helpers_data.denormalize_eval(dataset_or_img=normed_output_i,
        #                                                   data_normalisation=data_normalisation,
        #                                                   norm=norm_input, params=model.params, to_numpy=False)

        print(f'network output shape : {denormed_output_i.shape}')

        if 'sino' not in params:
            output_array = denormed_output_i.cpu().numpy()[:,0,:,:] if not params['full_sino'] else denormed_output_i.cpu().numpy()[0,:,:,:]
        else:
            if params['full_sino']:
                output_array = denormed_output_i.cpu().numpy()[0,:,:,:].transpose((1,0,2))
            else:
                # sino
                output_array = denormed_output_i.cpu().numpy()[:,0,4:124,:].transpose((1,0,2))
                # end sino
        print(f'final output shape : {output_array.shape}')
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
