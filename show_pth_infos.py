import torch
import click
import matplotlib.pyplot as plt

from DeepPVC import helpers,helpers_params, Model_instance


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth', 'lpth', multiple = True)
@click.option('--losses', is_flag = True, default = False)
def show_click(lpth, losses):
    show_pth(lpth, losses)


def show_pth(lpth, losses):
    lparams = []
    device = helpers.get_auto_device("cpu")

    dict_test={}

    for pth in lpth:
        nn = torch.load(pth,map_location=device)
        params = nn['params']
        helpers_params.check_params(params)
        params['jean_zay']=False
        lparams.append(params)
        ref=params['ref']

        print(params)

        model = Model_instance.ModelInstance(params=params, from_pth=pth,resume_training=False, device=device)
        model.load_model(pth_path=pth)
        model.switch_device("cpu")
        model.switch_eval()
        model.show_infos()

        if losses:
            model.plot_losses(save=False, wait=True, title=pth)
            dict_test[ref]=model.test_error

    if losses:
        fig_test,ax_test=plt.subplots()
        cm = plt.get_cmap('gist_rainbow')
        NUM_COLORS=len(dict_test.items())
        for i,(ref_i,test_i) in enumerate(dict_test.items()):
            print(test_i)
            ax_test.plot([e[0] for e in  test_i],[e[1] for e in  test_i], label=ref_i, color=cm(1.*i/NUM_COLORS))
        ax_test.legend()
        plt.show()

    # helpers_params.make_and_print_params_info_table(lparams)


if __name__ == '__main__':
    show_click()