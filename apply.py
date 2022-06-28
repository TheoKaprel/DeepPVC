import torch
import matplotlib.pyplot as plt
import numpy as np
import click
import itk
import glob


from DeepPVC import Pix2PixModel, helpers_data, helpers


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
    norm = params['norm']
    normalisation = params['data_normalisation']

    model = Pix2PixModel.PVEPix2PixModel(params=params, is_resume=False)
    model.load_model(pth)
    model.switch_device("cpu")
    model.switch_eval()
    model.show_infos()


    input_image = itk.imread(input)
    input_array = itk.array_from_image(input_image)
    input_array = np.expand_dims(input_array, axis=0)
    vSpacing = np.array(input_image.GetSpacing())
    vOffset = np.array(input_image.GetOrigin())



    normalized_input_tensor = helpers_data.normalize(dataset_or_img=input_array, normtype=normalisation, norm=norm,
                                                 to_torch=True, device='cpu')
    normalized_output_tensor = torch.zeros_like(normalized_input_tensor)

    N_proj = normalized_input_tensor.shape[1]

    for proj in range(N_proj):
        tensor_PVE_proj = normalized_input_tensor[:, proj, :, :]
        tensor_PVE_proj = tensor_PVE_proj[:, None, :, :]
        output_tensor_proj = model.test(tensor_PVE_proj)
        normalized_output_tensor[:,proj,:,:] = output_tensor_proj

    output_array = helpers_data.denormalize(dataset_or_img=normalized_output_tensor, normtype=normalisation, norm=norm, to_numpy=True)
    output_array = output_array[0,:,:,:]
    output_image = itk.image_from_array(output_array)
    output_image.SetSpacing(vSpacing)
    output_image.SetOrigin(vOffset)

    itk.imwrite(output_image, output_filename)
    print(f'Done! output at : {output_filename}')


if __name__ =='__main__':
    apply_click()
