import torch
import click
import matplotlib.pyplot as plt

from DeepPVC import helpers,helpers_params, Model_instance


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth', 'lpth', multiple = True)
@click.option('--losses', is_flag = True, default = False)
@click.option('--legend')
def show_click(lpth, losses, legend):
    show_pth(lpth, losses, legend)


def show_pth(lpth, losses, legend):
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
        if legend is None:
            legend = list(dict_test.keys())
        else:
            legend = legend.split(",")

        fig_test,ax_test=plt.subplots()
        # cm = plt.get_cmap('gist_rainbow')
        # NUM_COLORS=len(dict_test.items())
        colors = ['red', 'blue', 'orange', 'green', 'grey', 'violet']
        for i,(ref_i,test_i) in enumerate(dict_test.items()):
            print(test_i)
            ax_test.plot([e[0] for e in  test_i],[e[1] for e in  test_i],label=legend[i],
                         color=colors[i], linewidth = 2)
        ax_test.legend(fontsize = 18)
        ax_test.set_title('Test loss', fontsize = 18)
        ax_test.set_xlabel("Epochs", fontsize = 18)
        ax_test.set_ylabel("Test Loss (L1)", fontsize = 18)
        plt.show()

    # helpers_params.make_and_print_params_info_table(lparams)


if __name__ == '__main__':
    show_click()