import torch


def get_auto_device(device_mode):
    if device_mode=="auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_mode=="cpu":
        device = torch.device('cpu')
    elif device_mode in ["gpu", "cuda"]:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            print(f'WARNING : device was set to cuda but cuda is not available, so cpu...')

    print(f'Device : {device}')
    return device

def calc_2dConv_output_shape(input_dim, kernel_size, stride=1,padding=0, dilation=1):
    return int((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)