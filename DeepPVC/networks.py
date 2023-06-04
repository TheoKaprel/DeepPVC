import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.cuda.amp import custom_fwd

class DownSamplingBlock(nn.Module):
    def __init__(self, input_nc, output_nc,leaky_relu_val=0.2, kernel_size = (4,4), stride = (2,2), padding = 1,
                 norm="batch_norm", block="conv-relu-norm", res_unit=False):
        super(DownSamplingBlock, self).__init__()
        sequenceDownBlock = []
        splited_block = block.split('-')

        self.res_unit = res_unit
        if self.res_unit:
            self.res_conv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding = padding)

        down_done = False
        for elmt in splited_block:
            if (elmt=="conv" and down_done==False):
                sequenceDownBlock.append(nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding = padding))
                down_done=True
            elif (elmt=="conv" and down_done):
                sequenceDownBlock.append(nn.Conv2d(output_nc, output_nc, kernel_size=(3,3), stride=(1,1), padding=1))
            elif elmt=="relu":
                sequenceDownBlock.append(nn.LeakyReLU(leaky_relu_val, True))
            elif elmt=="norm":
                if norm == "batch_norm":
                    sequenceDownBlock.append(nn.BatchNorm2d(output_nc, track_running_stats=False))
                elif norm == "inst_norm":
                    sequenceDownBlock.append(nn.InstanceNorm2d(output_nc))

        self.sequenceDownBlock = nn.Sequential(*sequenceDownBlock)


    def forward(self, x):
        if self.res_unit:
            return self.res_conv(x)+self.sequenceDownBlock(x)
        else:
            return self.sequenceDownBlock(x)

class UpSamplingBlock(nn.Module):
    def __init__(self, input_nc, output_nc,leaky_relu_val=0.2,
                 norm="batch_norm", use_dropout = False, block="conv-relu-norm", res_unit = False):
        super(UpSamplingBlock, self).__init__()
        sequenceUpBlock = []
        splited_block = block.split('-')
        self.res_unit = res_unit
        if self.res_unit:
            self.res_conv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=(4,4), stride=(2,2), padding = (1,1))

        up_done = False
        for elmt in splited_block:
            if (elmt=="conv" and up_done==False):
                sequenceUpBlock.append(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=(4,4), stride=(2,2), padding = (1,1)))
                up_done=True
            elif (elmt=="conv" and up_done):
                sequenceUpBlock.append(nn.Conv2d(output_nc, output_nc, kernel_size=(3,3), stride=(1,1), padding=1))
            elif (elmt=="relu"):
                sequenceUpBlock.append(nn.LeakyReLU(leaky_relu_val,True))
            elif (elmt=="norm"):
                if norm == "batch_norm":
                    sequenceUpBlock.append(nn.BatchNorm2d(output_nc, track_running_stats=False))
                elif norm == "inst_norm":
                    sequenceUpBlock.append(nn.InstanceNorm2d(output_nc))

        self.sequenceUpBlock = nn.Sequential(*sequenceUpBlock)

        if use_dropout:
            self.use_dropout=True
            self.dropout = nn.Dropout(0.2,inplace=True)
        else:
            self.use_dropout=False

    def forward(self, x):
        if self.use_dropout:
            x = self.dropout(x)
        if self.res_unit:
            return self.res_conv(x)+self.sequenceUpBlock(x)
        else:
            return self.sequenceUpBlock(x)


class myminRelu(nn.ReLU):
    def __init__(self, vmin):
        super(myminRelu, self).__init__()
        self.vmin = vmin

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input - self.vmin, inplace=self.inplace) + self.vmin



class UNet(nn.Module):
    def __init__(self,input_channel, ngc,conv3d,init_feature_kernel,
                 output_channel,nb_ed_layers,generator_activation,
                 use_dropout,leaky_relu, norm, residual_layer=False, blocks="conv-relu-norm", ResUnet=False):
        super(UNet, self).__init__()

        if conv3d:
            self.conv3d=True
            self.threedConv = torch.nn.Conv3d(in_channels=1,out_channels=1,kernel_size=(3,3,3),stride=(1,1,1),padding=1)
            self.relu = torch.nn.ReLU()
        else:
            self.conv3d=False

        self.ResUnet = ResUnet

        init_feature_kernel_size = (int(init_feature_kernel),int(init_feature_kernel))
        init_feature_padding = int(init_feature_kernel/2)
        self.init_feature = nn.Conv2d(input_channel, ngc, kernel_size=init_feature_kernel_size, stride=(1,1), padding = init_feature_padding)

        self.nb_ed_layers = nb_ed_layers
        down_layers = []
        up_layers = []
        # Contracting layers :
        k = 1
        for _ in range(self.nb_ed_layers):
            down_layers.append(DownSamplingBlock(k * ngc,2 * k * ngc, norm = norm,leaky_relu_val=leaky_relu, block=blocks, res_unit=self.ResUnet))
            k = 2 * k
        self.down_layers = nn.Sequential(*down_layers)

        # Core layer
        # If any dropout layer is used, it is here
        up_layers.append(UpSamplingBlock(k * ngc, int(k/2) * ngc, norm=norm, use_dropout=use_dropout,leaky_relu_val=leaky_relu, block=blocks,res_unit=self.ResUnet))

        # Extracting layers :
        for _ in range(self.nb_ed_layers - 1):
            up_layers.append(UpSamplingBlock(k * ngc, int(k / 4) * ngc, norm = norm,leaky_relu_val=leaky_relu, block=blocks, res_unit=self.ResUnet))
            k = int( k / 2)

        self.up_layers = nn.Sequential(*up_layers)

        self.final_feature = nn.Conv2d(2 * ngc, output_channel, kernel_size=(3, 3), stride=(1, 1), padding = 1)

        self.residual_layer=residual_layer

        if generator_activation=="sigmoid":
            self.activation = nn.Sigmoid()
        elif generator_activation=="tanh":
            self.activation = nn.Tanh()
        elif generator_activation=="relu":
            self.activation = nn.ReLU()
        elif generator_activation=="softplus":
            self.activation = nn.Softplus()
        elif generator_activation=="none":
            self.activation = nn.Identity()

    def forward(self, x):
        if self.residual_layer:
            residual=x

        # 3D convolution
        if self.conv3d:
            x = x[:,None,:,:,:]
            x = self.threedConv(x)
            x = self.relu(x)
            x = x[:,0,:,:,:]
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

        # residual
        if self.residual_layer:
            y += residual[:,0:1,:,:]

        y = self.activation(y)
        # ----------------------------------------------------------
        return(y)

class NEncodingLayers(nn.Module):
    def __init__(self, input_channel, ndc,norm,leaky_relu, output_channel=1, blocks = "conv-relu-norm"):
        super(NEncodingLayers, self).__init__()

        # initial layer
        sequence = [nn.Conv2d(input_channel, ndc, kernel_size=(4, 4), stride=(2, 2), padding = 1), nn.LeakyReLU(leaky_relu, True)]

        #contracting lagers
        sequence += [DownSamplingBlock(ndc, 2 * ndc,leaky_relu_val=leaky_relu,norm=norm, block=blocks)]
        sequence += [DownSamplingBlock(2 * ndc, 4 * ndc,leaky_relu_val=leaky_relu,norm=norm, block=blocks)]

        sequence += [DownSamplingBlock(4*ndc, 8*ndc, stride = 1,leaky_relu_val=leaky_relu,norm=norm, block=blocks)]

        sequence += [nn.Conv2d(8 * ndc, output_channel, kernel_size=(4, 4), stride = (1, 1), padding=1)]
        self.model = nn.Sequential(*sequence)

    @custom_fwd
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




class AttentionBlock(nn.Module):
    def __init__(self, x_l_channels,x_l1_channels,int_channels):
        super(AttentionBlock, self).__init__()

        self.W_xl = nn.Sequential(
            nn.Conv2d(x_l_channels, int_channels, kernel_size=(1,1), stride=(1,1), padding=0, bias=True),
            nn.BatchNorm2d(int_channels)
        )

        self.W_xl1 = nn.Sequential(
            nn.Conv2d(x_l1_channels, int_channels,kernel_size=(1,1), stride=(1,1), padding=0, bias=True),
            nn.BatchNorm2d(int_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(int_channels, 1, kernel_size=(1,1), stride=(1,1), padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,x_l,x_l1):
        y_l = self.W_xl(x_l)
        y_l1 = self.W_xl1(x_l1)
        z = self.relu(y_l+y_l1)
        psi = self.psi(z)
        return x_l1*psi



class AttentionUNet(nn.Module):
    def __init__(self,input_channel, ngc,conv3d,init_feature_kernel,
                 output_channel,nb_ed_layers,generator_activation,
                 use_dropout,leaky_relu, norm, residual_layer=False):
        super(AttentionUNet, self).__init__()

        if conv3d:
            self.conv3d=True
            self.threedConv = torch.nn.Conv3d(in_channels=1,out_channels=1,kernel_size=(3,3,3),stride=(1,1,1),padding=1)
            self.relu = torch.nn.ReLU()
        else:
            self.conv3d=False

        init_feature_kernel_size = (int(init_feature_kernel),int(init_feature_kernel))
        init_feature_padding = int(init_feature_kernel/2)
        self.init_feature = nn.Conv2d(input_channel, ngc, kernel_size=init_feature_kernel_size, stride=(1,1), padding = init_feature_padding)

        self.nb_ed_layers = nb_ed_layers
        down_layers = []
        up_layers = []
        # Contracting layers :
        k = 1
        for _ in range(self.nb_ed_layers):
            down_layers.append(DownSamplingBlock(k * ngc,2 * k * ngc, norm = norm,leaky_relu_val=leaky_relu))
            k = 2 * k
        self.down_layers = nn.Sequential(*down_layers)

        # Core layer
        # If any dropout layer is used, it is here
        up_layers.append(UpSamplingBlock(k * ngc, int(k/2) * ngc, norm=norm, use_dropout=use_dropout,leaky_relu_val=leaky_relu))

        att_layers= []
        att_layers.append(AttentionBlock(x_l_channels=int(k/2) * ngc,
                                         x_l1_channels=int(k/2) * ngc,
                                         int_channels=int(k/4)*ngc))

        # Extracting layers :
        for _ in range(self.nb_ed_layers - 1):
            up_layers.append(UpSamplingBlock(k * ngc, int(k/4) * ngc, norm = norm,leaky_relu_val=leaky_relu))
            att_layers.append(AttentionBlock(x_l_channels= int(k/4*ngc),
                                         x_l1_channels=int(k/4*ngc),
                                         int_channels=int(k/8*ngc)))
            k = int(k / 2)

        self.up_layers = nn.Sequential(*up_layers)
        self.att_layers = nn.Sequential(*att_layers)

        self.final_feature = nn.Conv2d(2 * ngc, output_channel, kernel_size=(3, 3), stride=(1, 1), padding = 1)

        self.residual_layer=residual_layer

        if generator_activation=="sigmoid":
            self.activation = nn.Sigmoid()
        elif generator_activation=="tanh":
            self.activation = nn.Tanh()
        elif generator_activation=="relu":
            self.activation = nn.ReLU()
        elif generator_activation=="softplus":
            self.activation = nn.Softplus()
        elif generator_activation=="none":
            self.activation = nn.Identity()


    def forward(self, x):
        if self.residual_layer:
            residual=x

        # 3D convolution
        if self.conv3d:
            x = x[:,None,:,:,:]
            x = self.threedConv(x)
            x = self.relu(x)
            x = x[:,0,:,:,:]
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
            z = self.att_layers[l](xk[-l-2],y)
            xy = torch.cat([z,y],1)

        # ----------------------------------------------------------
        # Final feature extraction
        y = self.final_feature(xy) # output_channel

        # residual
        if self.residual_layer:
            y += residual[:,0:1,:,:]

        y = self.activation(y)
        # ----------------------------------------------------------
        return(y)