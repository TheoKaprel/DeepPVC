from prettytable import PrettyTable
from textwrap import fill


not_updatable_paramter_list_when_resume_training = ['dataset_path', 'training_batchsize', 'test_batchsize', 'training_prct',
                                                    'learning_rate','input_channels', 'hidden_channels_gen', 'hidden_channels_disc','optimizer',
                                                    'adv_loss', 'recon_loss','lambda_recon']




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
    """
    checks if
    - required params are in the 'params' dictionnary
    - types, min/max, closed option values
    - sets unspecified values to default (for unrequired parameters)
    - warning if unknown parameter
    """


    required = ['dataset_path','data_normalisation', 'n_epochs', 'learning_rate',
                'hidden_channels_gen', 'hidden_channels_disc',
                'generator_update', 'discriminator_update',
                'optimizer', 'device', 'lambda_recon', 'save_every_n_epoch']
    ballek = ['comment']

    automated = ['training_start_time', 'start_epoch','current_epoch', 'training_endtime','ref', 'output_folder', 'output_pth', 'start_pth', 'nb_training_data', 'nb_testing_data']

    default_params_values = [['test_dataset_path', params['dataset_path']] ,['training_batchsize', 5], ['test_batchsize', 5], ['input_channels',1], ["generator_activation", "sigmoid"],["generator_norm","batch_norm"], ['adv_loss','BCE'], ['recon_loss', 'L1'], ['show_every_n_epoch', 10], ["test_every_n_epoch", 10], ['training_duration', 0]]
    default_params = [param for param, value in default_params_values]

    for req in required:
        if (req not in params or req in [[], ""]):
            print(f'Error, the parameters "{req}" is required in {params}')
            exit(0)

    assert(type(params['n_epochs'])==int)
    assert((type(params['learning_rate'])==float) or (type(params['lambda_recon'])==int))
    assert(params['learning_rate']>0)
    assert(type(params['hidden_channels_gen'])==int)
    assert(params['hidden_channels_gen']>1)

    assert(type(params['optimizer'])==str)
    assert(params['optimizer'] in ["Adam"])

    assert((type(params['lambda_recon'])==float) or (type(params['lambda_recon'])==int))
    assert(params['lambda_recon']>=0)
    assert(type(params['save_every_n_epoch'])==int)
    assert(params['save_every_n_epoch']>0)


    for defparam, defvalue in default_params_values:
        if (defparam not in params) or (params[defparam] in [""]):
            params[defparam] = defvalue
            print(f'WARNING The {defparam} parameter has been automatically set to {defvalue}')

    for int_param in ['training_batchsize', 'test_batchsize', 'input_channels', 'generator_update', 'discriminator_update']:
        assert(type(params[int_param])==int)
        assert(params[int_param]>0)


    assert (params['device'] in ["cpu", "cuda", "auto"])
    assert(params['generator_activation'] in ["sigmoid", "tanh", "relu", "linear", "none", "relu_min"])
    assert (params['adv_loss'] in ["BCE"])
    assert (params['recon_loss'] in ["L1"])
    assert (params['data_normalisation'] in ["standard", "min_max", "min_max_1_1", "none"])
    assert (params['generator_norm'] in ["none", "batch_norm", "inst_norm"])

    for p in params:
        if p not in (required+automated+default_params+ballek):
            print(f'WARNING Unknown keynamed "{p}" in the params')
            if fatal_on_unknown:
                exit(0)




def make_and_print_params_info_table(lparams, mse=False):
    data_table = PrettyTable(align="l")
    data_table.title = "DATA PARAMETERS"
    data_table.field_names = ["Ref", "Train Dataset", "Test Dataset", "training_batchsize", "test_batchsize", "nb Train data", "nb Test data", "Comment"]

    train_table = PrettyTable()
    train_table.title = "MAIN INFORMATIONS"

    train_table_field_names = ["Ref","n_epochs", "data_normalisation", "generator_activation","generator_norm", "norm", "training_duration"]

    if mse:
        k = 1
        for dataset_filename,MSE_value in lparams[0]['MSE']:
            train_table_field_names.append(f'MSE on {dataset_filename.replace("/",".")}')
            k+=1
    train_table.field_names = train_table_field_names


    model_table = PrettyTable()
    model_table.title = "MODEL PARAMETERS"
    model_table.field_names = ["Ref", "learning rate", "input channels", "hidden channels generator", "hidden channels discriminator", "generator update", "discriminator update", "optimizer", "device", "adversarial loss", "recon loss", "lambda recon"]

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

        model_table.add_row([param['ref'], param['learning_rate'], param['input_channels'], param['hidden_channels_gen'], param['hidden_channels_disc'], param['generator_update'], param['discriminator_update'], param['optimizer'], param['device'], param['adv_loss'], param['recon_loss'], param['lambda_recon']])


    print(data_table)
    print('\n \n')
    print(train_table)
    print('\n \n')
    print(model_table)

