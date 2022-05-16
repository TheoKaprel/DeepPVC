


def update_params_user_option(params, user_params):
    """
    Update the dict 'params' with the options set by the user on the command line
    """
    for user_param in user_params:
        params[user_param[0]] = user_param[1]
