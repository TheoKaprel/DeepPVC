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
@click.option('--output', '-o', help = 'Output filename (mhd)')
@click.option("--device", default = "cpu")
def apply_click(pth,input,input_rec_fp, output, device):
    apply(pth, input,input_rec_fp, output_filename=output, device=device)


def apply(pth, input,input_rec_fp, output_filename, device):
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

    output_array = apply_to_input(input=input,input_rec_fp=input_rec_fp,params=params,device=device,model=model)


    input_image = itk.imread(input)
    vSpacing = np.array(input_image.GetSpacing())
    vOffset = np.array(input_image.GetOrigin())

    output_image = itk.image_from_array(output_array)
    output_image.SetSpacing(vSpacing)
    output_image.SetOrigin(vOffset)


    itk.imwrite(output_image, output_filename)
    print(f'Done! output at : {output_filename}')



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
        print(f'final output shape : {output_array.shape}')

        return output_array



if __name__ =='__main__':
    apply_click()
