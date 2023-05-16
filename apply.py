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
@click.option('--output', '-o', help = 'Output filename (mhd)')
def apply_click(pth,input, output):
    apply(pth, input, output_filename=output)


def apply(pth, input, output_filename):
    print(f'Apply the pth {pth} to the set of projections contained in {input}')

    device = helpers.get_auto_device("cpu")
    pth_file = torch.load(pth, map_location=device)
    params = pth_file['params']
    helpers_params.check_params(params)

    data_normalisation = params['data_normalisation']
    params['jean_zay']=False

    model = Model_instance.ModelInstance(params=params, from_pth=pth,resume_training=False,device=device)
    model.load_model(pth_path=pth)
    model.switch_device("cpu")
    model.switch_eval()
    model.show_infos()

    with torch.no_grad():

        input_with_channels = torch.tensor(helpers_data.load_image(filename=input, is_ref=False, type=None, params=params),
                                    device=device).float()
        print(f'input shape : {input_with_channels.shape}')

        norm_input = helpers_data.compute_norm_eval(dataset_or_img=input_with_channels, data_normalisation=data_normalisation)
        if data_normalisation!='none':
            print(f'norm shape : {norm_input[0].shape}')
        normed_input = helpers_data.normalize_eval(dataset_or_img=input_with_channels, data_normalisation=data_normalisation,
                                                     norm=norm_input, params=model.params, to_torch=False)

        normed_output_i = model.forward(normed_input.to(device))
        denormed_output_i = helpers_data.denormalize_eval(dataset_or_img=normed_output_i,
                                                          data_normalisation=data_normalisation,
                                                          norm=norm_input, params=model.params, to_numpy=False)

        print(f'network output shape : {denormed_output_i.shape}')
        output_array = denormed_output_i.cpu().numpy()[:,0,:,:]

        print(f'final output shape : {output_array.shape}')

        if input[-3:] in ["mhd", "mha"]:
            input_image = itk.imread(input)
            vSpacing = np.array(input_image.GetSpacing())
            vOffset = np.array(input_image.GetOrigin())
        else:
            vSpacing=np.array([4.41806,4.41806,1])
            vOffset=np.array([-280.5468,-280.54681,-59.5])

        output_image = itk.image_from_array(output_array)
        output_image.SetSpacing(vSpacing)
        output_image.SetOrigin(vOffset)


    itk.imwrite(output_image, output_filename)
    print(f'Done! output at : {output_filename}')


if __name__ =='__main__':
    apply_click()
