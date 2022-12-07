from prettytable import PrettyTable
from textwrap import fill


not_updatable_paramter_list_when_resume_training = ['training_batchsize', 'test_batchsize', 'training_prct',
                                                    'learning_rate','input_channels', 'hidden_channels_gen', 'hidden_channels_disc','optimizer',
                                                    'adv_loss', 'recon_loss','lambda_recon']

required = ['dataset_path', 'test_dataset_path','with_noise', 'data_normalisation', 'network', 'n_epochs', 'learning_rate',
            'input_channels','use_dropout','sum_norm','optimizer', 'device', 'lr_policy']

automated = ['training_start_time', 'start_epoch', 'current_epoch', 'training_endtime', 'ref', 'output_folder',
             'output_pth', 'start_pth', 'nb_training_data', 'nb_testing_data', 'norm']

ballek = ["comment"]

default_params_values = [["datatype", "mhd"], ['training_batchsize', 5],
                         ['test_batchsize', 5], ['save_every_n_epoch', 9999], ['show_every_n_epoch', 9999],
                         ["test_every_n_epoch", 9999], ['training_duration', 0], ["validation_norm", "L1"]]
default_params = [param for param, value in default_params_values]

required_pix2pix = ["nb_ed_layers", "hidden_channels_gen", "hidden_channels_disc",
                    "generator_activation", "generator_norm",
                    "adv_loss", "recon_loss", "lambda_recon", "generator_update", "discriminator_update"]

required_unet = ["nb_ed_layers", "hidden_channels_unet", "unet_activation", "unet_norm", "recon_loss"]

required_unet_denoiser_pvc = ["nb_ed_layers_denoiser","hidden_channels_unet_denoiser","unet_denoiser_activation","recon_loss_denoiser","unet_denoiser_norm",
                              "nb_ed_layers_pvc","hidden_channels_unet_pvc","unet_pvc_activation","recon_loss_pvc","unet_pvc_norm",
                              "denoiser_update","pvc_update"]



activation_functions = ["sigmoid", "tanh", "relu", "linear", "none", "relu_min"]
pre_layer_normalisations = ["batch_norm", "inst_norm", "none"]
losses = ["L1", "L2", "BCE"]
lr_policies = ["multiplicative"]



def update_params_user_option(params, user_params, is_resume):
    """
    Update the dict 'params' with the options set by the user on the command line
    """
    for user_param in user_params:
        if (is_resume and user_param[0] in not_updatable_paramter_list_when_resume_training):
            print(f'ERROR : parameter {user_param[0]} is not changeable when resuming training')
            exit(0)
        params[user_param[0]] = user_param[1]



def check_params(params, fatal_on_unknown=False):

    for req in required:
        if (req not in params or req in [[], ""]):
            print(f'ERROR: the parameters "{req}" is required in json param file')
            exit(0)

    for defparam, defvalue in default_params_values:
        if (defparam not in params) or (params[defparam] in [""]):
            params[defparam] = defvalue
            print(f'WARNING The {defparam} parameter has been automatically set to {defvalue}')


    if isinstance(params['dataset_path'], list)==False:
        params['dataset_path'] = [params['dataset_path']]
    for path in params['dataset_path']:
        assert(type(path)==str)
    if isinstance(params['test_dataset_path'], list)==False:
        params['test_dataset_path'] = [params['test_dataset_path']]
    for path in params['test_dataset_path']:
        assert(type(path)==str)

    assert (params['data_normalisation'] in ["global_standard","img_standard","global_0_1","img_0_1","img_mean","none"])
    assert (params["datatype"] in ["mhd", "mha"])
    assert (type(params['with_noise'])==bool)

    assert (params['network'] in ['pix2pix', 'unet', 'unet_denoiser_pvc', 'gan_denoiser_pvc'])


    int_param_list =  ['training_batchsize', 'test_batchsize','n_epochs', 'input_channels','save_every_n_epoch', 'show_every_n_epoch', 'test_every_n_epoch']
    for int_param in int_param_list:
        assert(type(params[int_param])==int)
        assert(params[int_param]>0)

    assert(type(params['use_dropout'])==bool)
    assert(type(params['sum_norm'])==bool)

    assert((type(params['learning_rate']) in [int, float]))
    assert(params['learning_rate']>0)

    assert(isinstance(params['lr_policy'], list))
    assert(params['lr_policy'][0] in lr_policies)
    assert(type(params['lr_policy'][1]) in [int, float])

    assert (params['optimizer'] in ["Adam"])
    assert (params['device'] in ["cpu", "cuda", "auto"])

    assert (params['validation_norm'] in losses)

    if params['network']=='pix2pix':
        check_params_pix2pix(params=params, fatal_on_unknown=fatal_on_unknown)
    elif params['network']=='unet':
        check_params_unet(params=params, fatal_on_unknown=fatal_on_unknown)
    elif params['network']=='unet_denoiser_pvc':
        check_params_unet_denoiser_pvc(params=params,fatal_on_unknown=fatal_on_unknown)



def check_params_pix2pix(params, fatal_on_unknown):

    for req in required_pix2pix:
        if (req not in params or req in [[], ""]):
            print(f'ERROR: the parameters "{req}" is required in json param file for Pix2Pix')
            exit(0)

    int_param_list = ["nb_ed_layers","hidden_channels_gen","hidden_channels_disc","generator_update","discriminator_update"]
    for int_param in int_param_list:
        assert(type(params[int_param])==int)
        assert(params[int_param]>0)

    assert(params['generator_activation'] in activation_functions)
    assert(params['generator_norm'] in pre_layer_normalisations)
    assert(params['adv_loss'] in losses)
    assert(params['recon_loss'] in losses)
    assert((type(params['lambda_recon']) in [int, float]))
    assert(params['lambda_recon'] >= 0)


    for p in params:
        if p not in (required+required_pix2pix+automated+default_params+ballek):
            if fatal_on_unknown:
                print(f'ERROR Unknown key named "{p}" in the params')
                exit(0)
            else:
                print(f'WARNING Unknown key named "{p}" in the params')


def check_params_unet(params, fatal_on_unknown):
    for req in required_unet:
        if (req not in params or req in [[], ""]):
            print(f'ERROR: the parameters "{req}" is required in json param file for UNet')
            exit(0)

    int_param_list = ["nb_ed_layers","hidden_channels_unet"]
    for int_param in int_param_list:
        assert(type(params[int_param])==int)
        assert(params[int_param]>0)

    assert(params['unet_activation'] in activation_functions)
    assert(params['unet_norm'] in pre_layer_normalisations)
    assert(params['recon_loss'] in losses)

    for p in params:
        if p not in (required+required_unet+automated+default_params+ballek):
            if fatal_on_unknown:
                print(f'ERROR Unknown key named "{p}" in the params')
                exit(0)
            else:
                print(f'WARNING Unknown key named "{p}" in the params')


def check_params_unet_denoiser_pvc(params, fatal_on_unknown):
    for req in required_unet_denoiser_pvc:
        if (req not in params or req in [[], ""]):
            print(f'ERROR: the parameters "{req}" is required in json param file for Unet Denoiser/PVC')
            exit(0)

    int_param_list = ["nb_ed_layers_denoiser", "hidden_channels_unet_denoiser", "nb_ed_layers_pvc", "hidden_channels_unet_pvc", "denoiser_update", "pvc_update"]
    for int_param in int_param_list:
        assert(type(params[int_param])==int)
        assert(params[int_param]>0)

    assert(params["unet_denoiser_activation"] in activation_functions)
    assert(params["unet_pvc_activation"] in activation_functions)

    assert(params['recon_loss_denoiser'] in losses)
    assert(params['recon_loss_pvc'] in losses)

    assert(params['unet_denoiser_norm'] in pre_layer_normalisations)
    assert(params['unet_pvc_norm'] in pre_layer_normalisations)

    for p in params:
        if p not in (required+required_unet_denoiser_pvc+automated+default_params+ballek):
            if fatal_on_unknown:
                print(f'ERROR Unknown key named "{p}" in the params')
                exit(0)
            else:
                print(f'WARNING Unknown key named "{p}" in the params')



def make_and_print_params_info_table(lparams):
    data_table = PrettyTable(align="l")
    data_table.title = "DATA PARAMETERS"
    data_table.field_names = ["Ref", "Train Dataset", "Test Dataset", "training_batchsize", "test_batchsize", "nb Train data", "nb Test data", "Comment"]

    train_table = PrettyTable()
    train_table.title = "MAIN INFORMATIONS"

    train_table_field_names = ["Ref","n_epochs", "data_normalisation", "generator_activation","generator_norm", "norm", "training_duration"]

    if 'MSE'in lparams[0]:
        mse = True
        for dataset_filename,MSE_value in lparams[0]['MSE']:
            train_table_field_names.append(f'MSE on {dataset_filename}')
    else:
        mse = False
    train_table.field_names = train_table_field_names


    model_table = PrettyTable()
    model_table.title = "MODEL PARAMETERS"
    model_table.field_names = ["Ref", "learning rate", "input channels", "hidden channels generator", "hidden channels discriminator","nb_ed_layers", "generator update", "discriminator update", "optimizer", "device", "adversarial loss", "recon loss", "lambda recon"]

    for param in lparams:
        ntrain_dataset = len(param['dataset_path'])
        ntest_dataset = len(param['test_dataset_path'])
        min_data = min((ntrain_dataset,ntest_dataset))

        data_table.add_row([param['ref'], fill(param['dataset_path'][0], width=50), fill(param['test_dataset_path'][0], width=50), param['training_batchsize'], param['test_batchsize'], param['nb_training_data'], param['nb_testing_data'], param['comment']])
        if min_data>0:
            for k in range(1,min_data):
                data_table.add_row(['', fill(param['dataset_path'][k], width=50),
                                    fill(param['test_dataset_path'][k], width=50), '','', '', '',''])
            if ntrain_dataset>min_data:
                for k in range(min_data, ntrain_dataset):
                    data_table.add_row(['', fill(param['dataset_path'][k], width=50),'', '', '', '', '', ''])
            elif ntest_dataset>min_data:
                for k in range(min_data, ntest_dataset):
                    data_table.add_row(['', '', fill(param['test_dataset_path'][k], width=50), '', '', '', '', ''])
        if mse:
            row = [param['ref'], param['n_epochs'], param['data_normalisation'], param['generator_activation'],param['generator_norm'], param['norm'], param['training_duration']]
            lmse = param['MSE']
            lmse_val = [k[1] for k in lmse]
            row = [*row, *lmse_val]
            train_table.add_row(row)
        else:
            train_table.add_row([param['ref'], param['n_epochs'], param['data_normalisation'], param['generator_activation'], param['generator_norm'], param['norm'], param['training_duration']])

        model_table.add_row([param['ref'], param['learning_rate'], param['input_channels'], param['hidden_channels_gen'], param['hidden_channels_disc'],param['nb_ed_layers'], param['generator_update'], param['discriminator_update'], param['optimizer'], param['device'], param['adv_loss'], param['recon_loss'], param['lambda_recon']])


    print(data_table)
    print('\n \n')
    print(train_table)
    print('\n \n')
    print(model_table)

