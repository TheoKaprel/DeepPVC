import matplotlib.pyplot as plt
import torch
import numpy as np
import click
import itk



from DeepPVC import Model_instance, helpers_data, helpers, helpers_params


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth') # 'path/to/saved/model.pth'
@click.option('--input', '-i')
@click.option('--input_rec_fp')
@click.option('--attmap_fp')
@click.option('--output', '-o', help = 'Output filename (mhd)')
@click.option("--device", default = "cpu")
def apply_click(pth,input,input_rec_fp,attmap_fp, output, device):
    output_image = apply(pth, input,input_rec_fp,attmap_fp, device=device)

    itk.imwrite(output_image, output)
    print(f'Done! output at : {output}')


def apply(pth, input,input_rec_fp,attmap_fp, device):
    print(f'Apply the pth {pth} to the set of projections contained in {input}')

    device = helpers.get_auto_device(device_mode=device)
    pth_file = torch.load(pth, map_location=device)
    params = pth_file['params']
    helpers_params.check_params(params)

    params['jean_zay']=False

    model = Model_instance.ModelInstance(params=params, from_pth=pth,resume_training=False,device=device)
    model.load_model(pth_path=pth)
    model.switch_device(device)
    model.switch_eval()
    model.show_infos()

    output_array = apply_to_input(input=input,input_rec_fp=input_rec_fp,attmap_fp=attmap_fp,params=params,device=device,model=model)


    input_image = itk.imread(input)
    vSpacing = np.array(input_image.GetSpacing())
    vOffset = np.array(input_image.GetOrigin())

    output_image = itk.image_from_array(output_array)
    output_image.SetSpacing(vSpacing)
    output_image.SetOrigin(vOffset)

    return output_image



def apply_to_input(input, input_rec_fp,attmap_fp, params, device, model):

    input_PVE_noisy_array = itk.array_from_image(itk.imread(input))
    input_rec_fp_array = itk.array_from_image(itk.imread(input_rec_fp)) if ((input_rec_fp is not None) and (params['with_rec_fp'])) else None
    attmap_fp_array = itk.array_from_image(itk.imread(attmap_fp)) if (attmap_fp is not None) else None

    with torch.no_grad():
        data_input = helpers_data.get_dataset_for_eval(params=params,
                                                       input_PVE_noisy_array=input_PVE_noisy_array,
                                                       input_rec_fp_array=input_rec_fp_array,
                                                       attmap_fp_array=attmap_fp_array)

        for key in data_input.keys():
            data_input[key] = data_input[key].to(device)

        print(f'input shape :  {[(k, v.shape) for (k,v) in data_input.items()]}')

        data_normalisation = params['data_normalisation']
        norm_input = helpers_data.compute_norm_eval(dataset_or_img=data_input, data_normalisation=data_normalisation)
        if (data_normalisation!='none'):
            print(f'norm : {norm_input}')
        data_input = helpers_data.normalize_eval(dataset_or_img=data_input, data_normalisation=data_normalisation,
                                                     norm=norm_input, params=model.params, to_torch=False)

        if params['inputs']=='projs':
            output = np.zeros((data_input['PVE_noisy'].shape[0], data_input['PVE_noisy'].shape[2], data_input['PVE_noisy'].shape[3]))
            batch_size = 4
            for i in range(data_input['PVE_noisy'].shape[0]//batch_size):
                print(i)
                batch={}
                for key in data_input.keys():
                    batch[key] = data_input[key][i*batch_size:(i+1)*batch_size,:,48:208,16:240]

                output[i*batch_size:(i+1)*batch_size,48:208,16:240] = model.forward(batch=batch)[:,0,:,:].cpu().numpy()

        elif params['inputs']=="full_sino":
            output = torch.zeros((data_input['PVE_noisy'].shape[1], data_input['PVE_noisy'].shape[2], data_input['PVE_noisy'].shape[3]))

            batch = {}
            for key in data_input.keys():
                batch[key] = data_input[key][:,:,48:208,16:240]

            output[:,48:208,16:240] = model.forward(batch)
        elif params['inputs']=="imgs":
            output=model.forward(data_input)

        output = helpers_data.denormalize_eval(dataset_or_img=output,
                                                          data_normalisation=data_normalisation,
                                                          norm=norm_input, params=model.params, to_numpy=False)

        print(f'network output shape : {output.shape}')

        output_array = helpers_data.back_to_input_format(params=params,output=output)
        print(f'final output shape : {output_array.shape}')


        return output_array



if __name__ =='__main__':
    apply_click()
