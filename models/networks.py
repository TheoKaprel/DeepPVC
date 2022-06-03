import torch
import torch.nn as nn
from torch.nn import init
import itk
import matplotlib.pyplot as plt


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
    def __init__(self, input_nc, output_nc, norm="batch_norm"):
        super(UpSamplingBlock, self).__init__()
        self.do_norm = (norm!="none")
        self.normtype = norm

        self.upConv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=(4,4), stride=(2,2), padding = (1,1))
        self.upRelu = nn.ReLU(True)

        if self.normtype=="batch_norm":
            self.upNorm = nn.BatchNorm2d(output_nc)
        elif self.normtype=="inst_norm":
            self.upNorm = nn.InstanceNorm2d(output_nc)


    def forward(self, x):
        x = self.upConv(x)
        x = self.upRelu(x)
        if self.do_norm :
            x = self.upNorm(x)
        return(x)


class UNetGenerator(nn.Module):
    """UNet shaped Generator
    Parameters :
    - input_channel : number of channels in input data
    - ngc : number of channels/features after first feature extraction
    - output_channel : number of channels desired for the output
    FIXME : ajouter options : nb_layers, dropout, normlayer...

    """
    def __init__(self,input_channel, ngc, output_channel,generator_activation, norm):
        super(UNetGenerator, self).__init__()
        self.init_feature = nn.Conv2d(input_channel, ngc, kernel_size=(3, 3), stride=(1, 1), padding = 1)

        self.down1 = DownSamplingBlock(ngc, 2 * ngc, norm=norm)
        self.down2 = DownSamplingBlock(2 * ngc, 4 * ngc, norm=norm)
        self.down3 = DownSamplingBlock(4 * ngc, 8 * ngc, norm=norm)
        self.down4 = DownSamplingBlock(8 * ngc, 16 * ngc, norm=norm)

        self.up1 = UpSamplingBlock(16 * ngc, 8 * ngc, norm=norm)
        self.up2 = UpSamplingBlock(16 * ngc, 4 * ngc, norm=norm)
        self.up3 = UpSamplingBlock(8 * ngc, 2 * ngc, norm=norm)
        self.up4 = UpSamplingBlock(4 * ngc, ngc, norm=norm)

        self.final_feature = nn.Conv2d(2 * ngc, output_channel, kernel_size=(3, 3), stride=(1, 1), padding = 1)

        if generator_activation=="sigmoid":
            self.activation = nn.Sigmoid()
        elif generator_activation=="tanh":
            self.activation = nn.Tanh()
        elif generator_activation=="relu":
            self.activation = nn.ReLU()
        elif generator_activation=="none":
            self.activation = nn.Identity()



    def forward(self, x):
        # ----------------------------------------------------------
        #first feature extraction
        x0 = self.init_feature(x) # nhc
        # ----------------------------------------------------------
        # Contracting layers :
        x1 = self.down1(x0) # 2*nhc
        x2 = self.down2(x1) # 4*nhc
        x3 = self.down3(x2) # 8*nhc
        x4 = self.down4(x3) # 16*nhc
        # ----------------------------------------------------------
        # Extracting layers :
        y4 = self.up1(x4)  # 8*nhc
        xy4 = torch.cat([x3,y4],1) #16*nhc
        y3 = self.up2(xy4) # 4*nhc
        xy3 = torch.cat([x2,y3],1) # 8*nhc
        y2 = self.up3(xy3) # 2*nhc
        xy2 = torch.cat([x1,y2],1) # 4*nhc
        y1 = self.up4(xy2) # nhc

        xy = torch.cat([x0,y1],1) # 2*nhc
        # ----------------------------------------------------------
        # Final feature extraction
        y = self.final_feature(xy) # output_channel
        y = self.activation(y)
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
        # sequence += [DownSamplingBlock(4 * ndc, 8 * ndc)]

        sequence += [DownSamplingBlock(4*ndc, 8*ndc, stride = 1)]

        sequence += [nn.Conv2d(8 * ndc, output_channel, kernel_size=(4, 4), stride = (1, 1), padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, A, B):
        x = torch.cat([A, B], 1)
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


