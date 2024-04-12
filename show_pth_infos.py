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
    dict_val={}

    for pth in lpth:
        nn = torch.load(pth,map_location=device)
        params = nn['params']
        helpers_params.check_params(params)
        params['jean_zay']=False
        lparams.append(params)
        ref=params['ref']+f'_{params["current_epoch"]}'

        print(params)

        model = Model_instance.ModelInstance(params=params, from_pth=pth,resume_training=False, device=device)
        model.load_model(pth_path=pth)
        model.switch_device("cpu")
        model.switch_eval()
        model.show_infos()

        if losses:
            model.plot_losses(save=False, wait=True, title=pth)
            dict_test[ref]=model.test_error
            # dict_val[ref]=model.val_error_MSE
            dict_val[ref]=model.val_error_MAE
            print(model.val_error_MSE)

    params_keys=list(lparams[0].keys())

    params_with_differences=[]
    for key in params_keys:
        all_keys=[]
        for par in lparams:
            if key in par:
                all_keys.append(par[key] if not type(par[key])==list else str(par[key]))
            else:
                all_keys.append('None')

        values=list(set(all_keys))
        if len(values)>1:
            params_with_differences.append(key)


    print(params_with_differences)
    for par in lparams:
        print(par['ref'])
        for key_diff in params_with_differences:
            if key_diff in par:
                print(f'{key_diff}: {par[key_diff]}')
            else:
                print(f'{key_diff}: NONE')
        print('--'*20)



    if losses:
        if legend is None:
            legend = list(dict_test.keys())
        else:
            legend = legend.split(",")

        fig_test,ax_test=plt.subplots()
        # cm = plt.get_cmap('gist_rainbow')
        # NUM_COLORS=len(dict_test.items())
        colors = ['red', 'blue', 'orange', 'green', 'grey', 'violet', 'black', 'pink', "cyan"]
        for i,(ref_i,test_i) in enumerate(dict_test.items()):
            print(test_i)
            ax_test.plot([e[0] for e in  test_i],[e[1] for e in  test_i],label=legend[i],
                         color=colors[i], linewidth = 2)
        for i,(ref_i,val_i) in enumerate(dict_val.items()):
            ax_test.plot([e[0] for e in  val_i],[e[1] for e in  val_i],label=f'{legend[i]} (val)',
                         color=colors[i], linewidth = 1, linestyle="dashed")
        ax_test.legend(fontsize = 18)
        ax_test.set_title('Test loss', fontsize = 18)
        ax_test.set_xlabel("Epochs", fontsize = 18)
        ax_test.set_ylabel("Test Loss (L1)", fontsize = 18)
        plt.show()


    # helpers_params.make_and_print_params_info_table(lparams)


if __name__ == '__main__':
    show_click()
