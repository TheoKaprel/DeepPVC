


def update_params_user_option(params, user_params):
    """
    Update the dict 'params' with the options set by the user on the command line
    """
    for user_param in user_params:
        params[user_param[0]] = user_param[1]



def check_params(params, fatal_on_unknown=False):

    required = ['dataset_path', 'n_epochs', 'learning_rate',
                'hidden_channels_gen', 'hidden_channels_disc',
                'optimizer', 'device', 'lambda_recon']

    automated = ['training_start_time', 'start_epoch', 'training_endtime', 'output_path']
    option = ['start_pth']
    default = ['training_batchsize', 'test_batchsize', 'training_prct', 'input_channels', 'display_step', 'adv_loss', 'recon_loss']

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
    assert (type(params['device']) == str)
    assert((type(params['lambda_recon'])==float) or (type(params['lambda_recon'])==int))
    assert(params['lambda_recon']>=0)

    if 'training_batchsize' not in params:
        params['training_batchsize'] = 5
        print('WARNING The training_batchsize parameter has been automatically set to 5')
    if 'test_batchsize' not in params:
        params['test_batchsize'] = 5
        print('WARNING The test_batchsize parameter has been automatically set to 5')
    if 'training_prct' not in params:
        params['training_prct'] = 0.2
        print('WARNING The training_prct parameter has been automatically set to 20%')
    if 'input_channels' not in params:
        params['input_channels'] = 1
        print('WARNING The input_channels parameter has been automatically set to 1')
    if 'display_step' not in params:
        params['display_step'] = 30
        print('WARNING The display_step parameter has been automatically set to 30')
    if 'adv_loss' not in params:
        params['adv_loss'] = "BCE"
        print('WARNING The adv_loss parameter has been automatically set to Adam')
    if 'recon_loss' not in params:
        params['recon_loss'] = "L1"
        print('WARNING The recon_loss parameter has been automatically set to Adam')

    for int_param in ['training_batchsize', 'test_batchsize', 'input_channels', 'display_step']:
        assert(type(params[int_param])==int)
        assert(params[int_param]>0)

    assert (type(params['training_prct']) == float)
    assert (params['training_prct'] >= 0)
    assert (params['adv_loss'] in ["BCE"])
    assert (params['recon_loss'] in ["L1"])

    for p in params:
        if p not in (required+automated+option+default):
            print(f'WARNING Unknown keynamed "{p}" in the params')
            if fatal_on_unknown:
                exit(0)





