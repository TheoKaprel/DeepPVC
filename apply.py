import torch
import numpy as np
import click
import itk



from DeepPVC import Models, helpers_data, helpers, helpers_params


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth') # 'path/to/saved/model.pth'
@click.option('--input', '-i')
@click.option('--output_filename', '-o', help = 'Output filename (mhd)')

def apply_click(pth,input, output_filename):
    apply(pth, input, output_filename)


def apply(pth, input, output_filename):
    print(f'Apply the pth {pth} to the set of projections contained in {input}')

    device = helpers.get_auto_device("auto")
    pth_file = torch.load(pth, map_location=device)
    params = pth_file['params']
    helpers_params.check_params(params)

    norm = params['norm']
    normalisation = params['data_normalisation']

    model = Models.ModelInstance(params=params, from_pth=pth)

    model.switch_device("cpu")
    model.switch_eval()
    model.show_infos()

    with torch.no_grad():
        input_image = itk.imread(input)
        input_array = itk.array_from_image(input_image)
        input_array = np.expand_dims(input_array,axis=1)
        print(input_array.shape)

        vSpacing = np.array(input_image.GetSpacing())
        vOffset = np.array(input_image.GetOrigin())

        normalized_input_tensor = helpers_data.normalize(dataset_or_img=input_array, normtype=normalisation, norm=norm,
                                                     to_torch=True, device='cpu')

        normalized_output_tensor = model.forward(normalized_input_tensor)

        output_array = helpers_data.denormalize(dataset_or_img=normalized_output_tensor, normtype=normalisation, norm=norm, to_numpy=True)[:,0,:,:]
        output_image = itk.image_from_array(output_array)
        output_image.SetSpacing(vSpacing)
        output_image.SetOrigin(vOffset)

    itk.imwrite(output_image, output_filename)
    print(f'Done! output at : {output_filename}')


if __name__ =='__main__':
    apply_click()
