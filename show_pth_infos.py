import torch
import click
import matplotlib.pyplot as plt

from DeepPVC import helpers,helpers_params, Model_instance
from tabulate import tabulate
from torchscan import summary

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
    dict_train={}

    for pth in lpth:
        nn = torch.load(pth,map_location=device)
        params = nn['params']
        helpers_params.check_params(params)
        params['jean_zay']=False
        # lparams.append(params)
        ref=params['ref']+f'_{params["current_epoch"]}'

        print(params)

        model = Model_instance.ModelInstance(params=params, from_pth=pth,resume_training=False, device=device)
        model.load_model(pth_path=pth)
        model.switch_device("cpu")
        model.switch_eval()
        model.show_infos()
        # summary(module = model.UNet,input_shape=(3,128,80,112),receptive_field=True)

        params['nb_params']= model.nb_params
        lparams.append(params)

        if losses:
            model.plot_losses(save=False, wait=True, title=pth)
            dict_test[ref]=model.test_error
            dict_val[ref]=model.val_error_MSE
            dict_val[ref]=model.val_error_MAE
            dict_train[ref]=model.unet_losses
            print(model.val_error_MSE)

    params_keys=list(lparams[-1].keys())

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

    tab = []
    print(params_with_differences)
    l_not_intab =  ["validation_dataset_path__", "output_folder", "dataset_path", "test_dataset_path"]
    for par in lparams:
        t = []
        print(par['ref'])
        for key_diff in params_with_differences:
            if key_diff in par:
                print(f'{key_diff}: {par[key_diff]}')
                if key_diff not in l_not_intab:
                    t.append(par[key_diff])
            else:
                print(f'{key_diff}: NONE')
                if key_diff not in  l_not_intab:
                    t.append("NONE")

        print('--'*20)
        tab.append(t)
    tab_head = [p for p in params_with_differences if p not in l_not_intab]

    print(tabulate(tab,headers=tab_head))



    if losses:
        if legend is None:
            legend = list(dict_test.keys())
        else:
            legend = legend.split(",")

        fig_train,ax_train=plt.subplots()
        colors = ['red', 'blue', 'orange', 'green', 'grey', 'violet', 'black', 'pink', "cyan", "gold", "blueviolet", 'grey', 'magenta']
        for i,(ref_i,train_i) in enumerate(dict_train.items()):
            ax_train.plot(train_i,label=legend[i],
                         color=colors[i], linewidth = 2)
        ax_train.legend(fontsize = 18)
        ax_train.set_title('Training loss', fontsize = 18)
        ax_train.set_xlabel("Epochs", fontsize = 18)
        ax_train.set_ylabel("Training Losses", fontsize = 18)


        fig_test,ax_test=plt.subplots()
        # cm = plt.get_cmap('gist_rainbow')
        # NUM_COLORS=len(dict_test.items())
        colors = ['red', 'blue', 'orange', 'green', 'grey', 'violet', 'black', 'pink', "cyan", "gold", "blueviolet", 'grey', 'magenta']
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

        fig_train_test,ax_train_test=plt.subplots()
        # colors = ['red', 'blue', 'orange', 'green', 'grey', 'violet', 'black', 'pink', "cyan", "gold", "blueviolet"]
        # colors = ['orange', 'blue', 'orange', 'green', 'grey', 'violet', 'black', 'pink', "cyan", "gold", "blueviolet"]
        for i,(ref_i,test_i) in enumerate(dict_test.items()):
            print(test_i)
            ax_train_test.plot([e[0] for e in  test_i],[e[1] for e in  test_i],label=f'Validation',
                         color=colors[i], linewidth = 2)
        for i,(ref_i,train_i) in enumerate(dict_train.items()):
            ax_train_test.plot([_ for _ in range(len(train_i))], train_i,label=f'Training',
                         color=colors[i], linewidth = 1, linestyle="dashed")
        ax_train_test.legend(fontsize = 18)
        # ax_train_test.set_title('losses', fontsize = 18)
        ax_train_test.set_xlabel("Epochs", fontsize = 18)
        ax_train_test.set_ylabel("L1 Losses", fontsize = 18)
        plt.show()


    # helpers_params.make_and_print_params_info_table(lparams)


if __name__ == '__main__':
    show_click()
