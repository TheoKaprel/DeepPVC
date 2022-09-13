import torch
import click


from DeepPVC import helpers,helpers_params


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth', 'lpth', multiple = True)
def show_click(lpth):
    show_pth(lpth)


def show_pth(lpth):
    lparams = []
    device = helpers.get_auto_device("cpu")

    for pth in lpth:
        nn = torch.load(pth,map_location=device)
        params = nn['params']
        helpers_params.check_params(params)
        lparams.append(params)


    helpers_params.make_and_print_params_info_table(lparams)


if __name__ == '__main__':
    show_click()