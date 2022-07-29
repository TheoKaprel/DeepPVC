import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

class DownSamplingBlock(nn.Module):
    """ DownSampling Block = One Subblock of the left part of a Unet --> Encoding
    Parameters :
    - input_nc : number of channels in the input
    - output_nc : number of desired channels in the output

    For an input X of dimension (N,N, input_nc) the output of this block is dimension (N/2,N/2,output_nc)
    car
    si dim(x) = (I,I) et y = Conv2D(x)
    alors dim(y) = (I - kernel_size + 2*padding)/stride +1 = (I-4+2)/2 +1 = I/2
    """
    def __init__(self, input_nc, output_nc, kernel_size = (4,4), stride = (2,2), padding = 1, norm="batch_norm"):
        super(DownSamplingBlock, self).__init__()

        self.do_norm = (norm!="none")
        self.normtype = norm
        self.downConv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding = padding)
        self.downRelu = nn.LeakyReLU(0.2, True)

        if self.normtype=="batch_norm":
            self.downNorm = nn.BatchNorm2d(output_nc)
        elif self.normtype=="inst_norm":
            self.downNorm = nn.InstanceNorm2d(output_nc)

    def forward(self, x):
        x = self.downConv(x)
        x = self.downRelu(x)
        if self.do_norm:
            x = self.downNorm(x)
        return(x)

class UpSamplingBlock(nn.Module):
    """ Up-sampling Block = One block of the right part of a Unet  --> Decoding
    Parameters :
    - input_nc : number of channels in the input
    - output_nc : number of desired channels in the output
    - norm : wether or not to normalize the output

    For an input X of dimension (N,N, input_nc) the output of this block is dimension (2N,2N,output_nc)
    car
    si dim(x) = (I,I) et y = ConvTranspose2d(x)
    alors dim(y) = (I-1)stride - 2*padding + kernel_size =  (I-1)*2 - 2 * 1 + 4 = 2 I
    """
    def __init__(self, input_nc, output_nc, norm="batch_norm", use_dropout = False):
        super(UpSamplingBlock, self).__init__()
        self.do_norm = (norm!="none")
        self.normtype = norm
        self.use_dropout = use_dropout

        self.upConv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=(4,4), stride=(2,2), padding = (1,1))
        self.upRelu = nn.ReLU(True)

        if self.normtype=="batch_norm":
            self.upNorm = nn.BatchNorm2d(output_nc)
        elif self.normtype=="inst_norm":
            self.upNorm = nn.InstanceNorm2d(output_nc)

        if self.use_dropout:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.upConv(x)

        if self.use_dropout:
            x = self.dropout(x)

        x = self.upRelu(x)
        if self.do_norm :
            x = self.upNorm(x)
        return(x)


class myminRelu(nn.ReLU):
    def __init__(self, vmin):
        super(myminRelu, self).__init__()
        self.vmin = vmin

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input - self.vmin, inplace=self.inplace) + self.vmin



def mySumNormAtivationFct(x0,x):
    sum_x = x.sum(dim=(2, 3), keepdims=True)
    x = x / sum_x
    sum_x0 = x0.sum(dim=(2, 3), keepdims=True)
    x = x * sum_x0
    return x



class UNetGenerator(nn.Module):
    """UNet shaped Generator
    Parameters :
    - input_channel : number of channels in input data
    - ngc : number of channels/features after first feature extraction
    - output_channel : number of channels desired for the output
    - nb_ed_layers
    - generator_activation
    - norm
    - vmin = None
    FIXME : ajouter options :  dropout
    """
    def __init__(self,input_channel, ngc, output_channel,nb_ed_layers,generator_activation,use_dropout,sum_norm, norm, vmin = None):
        super(UNetGenerator, self).__init__()
        self.init_feature = nn.Conv2d(input_channel, ngc, kernel_size=(3, 3), stride=(1, 1), padding = 1)

        self.nb_ed_layers = nb_ed_layers
        down_layers = []
        up_layers = []
        # Contracting layers :
        k = 1
        for _ in range(self.nb_ed_layers):
            down_layers.append(DownSamplingBlock(k * ngc,2 * k * ngc, norm = norm))
            k = 2 * k
        self.down_layers = nn.Sequential(*down_layers)

        # Core layer
        # If any dropout layer is used, it is here
        up_layers.append(UpSamplingBlock(k * ngc, int(k/2) * ngc, norm=norm, use_dropout=use_dropout))

        # Extracting layers :
        for _ in range(self.nb_ed_layers - 1):
            up_layers.append(UpSamplingBlock(k * ngc, int(k / 4) * ngc, norm = norm))
            k = int( k / 2)

        self.up_layers = nn.Sequential(*up_layers)

        self.final_feature = nn.Conv2d(2 * ngc, output_channel, kernel_size=(3, 3), stride=(1, 1), padding = 1)


        if generator_activation=="sigmoid":
            self.activation = nn.Sigmoid()
        elif generator_activation=="tanh":
            self.activation = nn.Tanh()
        elif generator_activation=="relu":
            self.activation = nn.ReLU()
        elif generator_activation=="none":
            self.activation = nn.Identity()
        elif generator_activation=='relu_min':
            self.activation = myminRelu(vmin)

        self.sum_norm = sum_norm



    def forward(self, x):
        x = x.float()
        # ----------------------------------------------------------
        #first feature extraction
        x0 = self.init_feature(x) # nhc
        # ----------------------------------------------------------
        # Contracting layers :
        xk  = [x0]
        for l in range(self.nb_ed_layers):
            xk.append(self.down_layers[l](xk[-1]))
        # ----------------------------------------------------------
        # Extracting layers :
        xy = xk[-1]
        for l in range(self.nb_ed_layers):
            y = self.up_layers[l](xy)
            xy = torch.cat([xk[-l-2],y],1)

        # ----------------------------------------------------------
        # Final feature extraction
        y = self.final_feature(xy) # output_channel
        y = self.activation(y)
        if self.sum_norm:
            y = mySumNormAtivationFct(x, y)
        # ----------------------------------------------------------
        return(y)

class NLayerDiscriminator(nn.Module):
    """N convolutionnal layers Discriminator
    Parameters :
    - input_channel : number of channels in input data
    - ndc : number of channels/features after first feature extraction
    - output_channel : number of channels desired for the output
    FIXME : ajouter options : nb_layers, dropout, normlayer
    """
    def __init__(self, input_channel, ndc, output_channel=1):
        super(NLayerDiscriminator, self).__init__()

        # initial layer
        sequence = [nn.Conv2d(input_channel, ndc, kernel_size=(4, 4), stride=(2, 2), padding = 1), nn.LeakyReLU(0.2, True)]

        #contracting lagers
        sequence += [DownSamplingBlock(ndc, 2 * ndc)]
        sequence += [DownSamplingBlock(2 * ndc, 4 * ndc)]

        sequence += [DownSamplingBlock(4*ndc, 8*ndc, stride = 1)]

        sequence += [nn.Conv2d(8 * ndc, output_channel, kernel_size=(4, 4), stride = (1, 1), padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, A, B):
        x = torch.cat([A.float(), B.float()], 1)
        return self.model(x)

    def get_receptive_field(self):
        R = 1
        prod_S = 1
        for layer in self.model.children():
            if isinstance(layer, nn.Conv2d):
                R = R + (layer.kernel_size[0] - 1)*prod_S
                prod_S = prod_S * layer.stride[0]
            elif isinstance(layer, DownSamplingBlock):
                for sublayer in layer.children():
                    if isinstance(sublayer, nn.Conv2d):
                        R = R + (sublayer.kernel_size[0] - 1) * prod_S
                        prod_S = prod_S * sublayer.stride[0]
        return R
