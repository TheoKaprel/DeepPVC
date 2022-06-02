
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


    required = ['dataset_path', 'n_epochs', 'learning_rate',
                'hidden_channels_gen', 'hidden_channels_disc',
                'generator_update', 'discriminator_update',
                'optimizer', 'device', 'lambda_recon', 'save_every_n_epoch']

    automated = ['training_start_time', 'start_epoch','current_epoch', 'training_endtime', 'output_path', 'start_pth', 'nb_training_data', 'nb_testing_data']

    default_params_values = [['test_dataset_path', params['dataset_path']] ,['training_batchsize', 5], ['test_batchsize', 5], ['training_prct',0.2],['data_normalisation','sum'], ['input_channels',1], ["generator_activation", "sigmoid"], ['adv_loss','BCE'], ['recon_loss', 'L1'], ['show_every_n_epoch', 10], ["test_every_n_epoch", 10]]
    default_params = [param for param, value in default_params_values]

    for req in required:
        if req not in params:
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
    assert (type(params['training_prct']) == float)
    assert (params['training_prct'] >= 0)
    assert(params['generator_activation'] in ["sigmoid", "tanh", "relu", "linear", "none"])
    assert (params['adv_loss'] in ["BCE"])
    assert (params['recon_loss'] in ["L1"])
    assert (params['data_normalisation'] in ["standard", "min_max", "none"])

    for p in params:
        if p not in (required+automated+default_params):
            print(f'WARNING Unknown keynamed "{p}" in the params')
            if fatal_on_unknown:
                exit(0)





