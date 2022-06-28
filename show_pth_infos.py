import torch
import click

from DeepPVC import Pix2PixModel, helpers


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('pth')
def show_click(pth):
    show_pth(pth)


def show_pth(pth):

    device = helpers.get_auto_device("auto")
    nn = torch.load(pth,map_location=device)
    print('-'*80)
    print(f'Showing informations about file {pth} : ')

    print(f'Saved at : {nn["saving_date"]}')
    print(f'Number of Epochs : {nn["epoch"]}')

    params = nn['params']

    pix2pix = Pix2PixModel.PVEPix2PixModel(params, is_resume=True, pth=pth)
    pix2pix.show_infos()


    print('-'*80)



if __name__ == '__main__':
    show_click()