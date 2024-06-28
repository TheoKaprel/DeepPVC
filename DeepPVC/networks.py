import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.cuda.amp import custom_fwd
from torch.nn.modules.utils import _triple
from torch.nn import init

from . import networks_attention_cbam
from .networks_vision_transformer import VisionTransformer,get_3DReg_config


def init_weights(net, init_type='normal', mean = 0.0, init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        # print(classname)
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, mean, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, mean)
        elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, mean)

    if init_type!="none":
        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>


class DownSamplingBlock(nn.Module):
    def __init__(self, input_nc, output_nc,leaky_relu_val=0.2, kernel_size = (3,3), stride = (2,2), padding = 1,
                 norm="batch_norm", block="conv-relu-norm", res_unit=False,dim=2, last=False):
        super(DownSamplingBlock, self).__init__()
        sequenceDownBlock = []
        splited_block = block.split('-')

        if dim==2:
            self.dim = 2
            conv = nn.Conv2d
            stride_one = (1,1)
            pool = nn.MaxPool2d

            if norm=="batch_norm":
                norm_layer = nn.BatchNorm2d
            elif norm=="inst_norm":
                norm_layer = nn.InstanceNorm2d
            else:
                norm_layer = nn.Identity

        elif dim==3:
            self.dim=3
            conv=nn.Conv3d
            kernel_size = kernel_size+(kernel_size[0],)
            stride = stride+(stride[0],)
            stride_one = (1,1,1)
            pool = nn.MaxPool3d

            if norm=="batch_norm":
                norm_layer = nn.BatchNorm3d
            elif norm=="inst_norm":
                norm_layer = nn.InstanceNorm3d
            else:
                norm_layer = nn.Identity

        self.res_unit = res_unit
        if self.res_unit:
            self.res_conv = conv(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding = padding)

        first_conv = True
        for elmt in splited_block:
            if (elmt=='downconv'):
                sequenceDownBlock.append(conv(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding))
                first_conv = True
            elif (elmt=="conv"):
                if first_conv:
                    sequenceDownBlock.append(conv(input_nc, output_nc, kernel_size=kernel_size, stride=stride_one, padding=padding))
                    first_conv=False
                else:
                    sequenceDownBlock.append(conv(output_nc, output_nc, kernel_size=kernel_size, stride=stride_one, padding=padding))
            elif elmt=="relu":
                sequenceDownBlock.append(nn.LeakyReLU(leaky_relu_val,inplace=False))
            elif elmt=="prelu":
                sequenceDownBlock.append(nn.PReLU())
            elif (elmt=='pool' and last==False):
                sequenceDownBlock.append(pool(kernel_size=kernel_size,stride=stride, padding=padding))
            elif elmt=="norm":
                sequenceDownBlock.append(norm_layer(output_nc))

        self.sequenceDownBlock = nn.Sequential(*sequenceDownBlock)


    def forward(self, x):
        if self.res_unit:
            return self.res_conv(x)+self.sequenceDownBlock(x)
        else:
            return self.sequenceDownBlock(x)

class UpSamplingBlock(nn.Module):
    def __init__(self, input_nc, output_nc,leaky_relu_val=0.2,
                 norm="batch_norm", use_dropout = False, block="conv-relu-norm", res_unit = False,dim=2):
        super(UpSamplingBlock, self).__init__()
        sequenceUpBlock = []
        splited_block = block.split('-')

        if dim==2:
            self.dim = 2
            conv = nn.Conv2d
            convT = nn.ConvTranspose2d
            kernel_size = (3,3)
            stride=(2,2)
            stride_one = (1,1)
            padd = (1,1)
            outpadd=(1,1)
            if norm=="batch_norm":
                norm_layer = nn.BatchNorm2d
            elif norm=="inst_norm":
                norm_layer = nn.InstanceNorm2d
            else:
                norm_layer = nn.Identity
        elif dim==3:
            self.dim=3
            conv=nn.Conv3d
            convT = nn.ConvTranspose3d
            kernel_size = (3,3,3)
            stride=(2,2,2)
            padd = (1,1,1)
            outpadd=(1,1,1)
            stride_one = (1,1,1)
            if norm=="batch_norm":
                norm_layer = nn.BatchNorm3d
            elif norm=="inst_norm":
                norm_layer = nn.InstanceNorm3d
            else:
                norm_layer = nn.Identity

        self.res_unit = res_unit
        if self.res_unit:
            if splited_block[0]=="convT":
                self.res_conv = convT(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding = padd, output_padding=outpadd)
            elif splited_block[0]=="upconv":
                res_conv = []
                res_conv.append(nn.Upsample(scale_factor=2))
                res_conv.append(conv(input_nc, output_nc, kernel_size=kernel_size, stride=stride_one, padding=padd))
                self.res_conv = nn.Sequential(*res_conv)

        for elmt in splited_block:
            if (elmt=="convT"):
                sequenceUpBlock.append(convT(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding = padd, output_padding=outpadd))
            elif (elmt=="conv"):
                sequenceUpBlock.append(conv(output_nc, output_nc, kernel_size=kernel_size, stride=stride_one, padding=padd))
            elif (elmt=="upconv"):
                sequenceUpBlock.append(nn.Upsample(scale_factor=2))
                sequenceUpBlock.append(conv(input_nc, output_nc, kernel_size=kernel_size, stride=stride_one, padding=padd))
            elif (elmt=="relu"):
                sequenceUpBlock.append(nn.LeakyReLU(leaky_relu_val,inplace=False))
            elif (elmt=="norm"):
                sequenceUpBlock.append(norm_layer(output_nc))

        self.sequenceUpBlock = nn.Sequential(*sequenceUpBlock)

        if use_dropout:
            self.use_dropout=True
            self.dropout = nn.Dropout(0.2,inplace=False)
        else:
            self.use_dropout=False

    def forward(self, x):
        if self.use_dropout:
            x = self.dropout(x)
        if self.res_unit:
            return self.res_conv(x)+self.sequenceUpBlock(x)
        else:
            return self.sequenceUpBlock(x)


class UpSamplingBlockBis(nn.Module):
    def __init__(self, input_nc, output_nc,leaky_relu_val=0.2,
                 norm="batch_norm", use_dropout = False, block="conv-relu-norm", res_unit = False,dim=2):
        super(UpSamplingBlockBis, self).__init__()
        sequenceUpBlock = []
        splited_block = block.split('-')

        if dim==2:
            self.dim = 2
            conv = nn.Conv2d
            convT = nn.ConvTranspose2d
            kernel_size = (3,3)
            stride=(2,2)
            stride_one = (1,1)
            padd = (1,1)
            outpadd=(1,1)
            if norm=="batch_norm":
                norm_layer = nn.BatchNorm2d
            elif norm=="inst_norm":
                norm_layer = nn.InstanceNorm2d
            else:
                norm_layer = nn.Identity
        elif dim==3:
            self.dim=3
            conv=nn.Conv3d
            kernel_size = (3,3,3)
            padd = (1,1,1)
            outpadd=(1,1,1)
            stride_one = (1,1,1)
            if norm=="batch_norm":
                norm_layer = nn.BatchNorm3d
            elif norm=="inst_norm":
                norm_layer = nn.InstanceNorm3d
            else:
                norm_layer = nn.Identity

        self.res_unit = res_unit
        if self.res_unit:
            if splited_block[0]=="convT":
                self.res_conv = convT(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding = padd, output_padding=outpadd)
            elif splited_block[0]=="upconv":
                res_conv = []
                res_conv.append(nn.Upsample(scale_factor=2))
                res_conv.append(conv(input_nc, output_nc, kernel_size=kernel_size, stride=stride_one, padding=padd))
                self.res_conv = nn.Sequential(*res_conv)

        self.up = nn.Upsample(scale_factor=2)
        self.up_conv = nn.Conv3d(input_nc, input_nc//2,(3,3,3),(1,1,1),(1,1,1))

        first = True
        for elmt in splited_block:
            if (elmt=="conv"):
                if first==True:
                    sequenceUpBlock.append(conv(input_nc, output_nc, kernel_size=kernel_size, stride=stride_one, padding=padd))
                    first=False
                else:
                    sequenceUpBlock.append(conv(output_nc, output_nc, kernel_size=kernel_size, stride=stride_one, padding=padd))
            elif (elmt=="relu"):
                sequenceUpBlock.append(nn.LeakyReLU(leaky_relu_val,inplace=False))
            elif (elmt=="prelu"):
                sequenceUpBlock.append(nn.PReLU())
            elif (elmt=="norm"):
                sequenceUpBlock.append(norm_layer(output_nc))

        self.sequenceUpBlock = nn.Sequential(*sequenceUpBlock)

        if use_dropout:
            self.use_dropout=True
            self.dropout = nn.Dropout(0.2,inplace=False)
        else:
            self.use_dropout=False

    def forward(self, x,y):

        if self.use_dropout:
            x = self.dropout(x)
        up_x = self.up(x)
        up_x = self.up_conv(up_x)
        cat_xy = torch.cat((up_x,y),1)
        if self.res_unit:
            return self.res_conv(x)+self.sequenceUpBlock(x)
        else:
            return self.sequenceUpBlock(cat_xy)

class PathsBlock(nn.Module):
    def __init__(self, input_channels, nb_channels_per_paths, nconv,kernel_size,stride, padding):
        super(PathsBlock, self).__init__()
        self.input_channels = input_channels

        init_paths = []
        for _ in range(input_channels):
            path = [torch.nn.Conv3d(1, nb_channels_per_paths, kernel_size=kernel_size, stride=stride,
                                 padding=padding),
                            nn.InstanceNorm3d(nb_channels_per_paths),
                            nn.LeakyReLU(0.02)]
            for __ in range(nconv-1):
                path = path + [torch.nn.Conv3d(nb_channels_per_paths, nb_channels_per_paths, kernel_size=kernel_size, stride=stride,
                                 padding=padding),
                               nn.InstanceNorm3d(nb_channels_per_paths),
                               nn.LeakyReLU(0.02)]
            init_paths.append(nn.Sequential(*path))
        self.init_paths = nn.Sequential(*init_paths)
    def forward(self, input):
        inputs = [input[:,k:k+1,:,:,:] for k in range(self.input_channels)]
        outputs = [self.init_paths[k](input_k) for k,input_k in enumerate(inputs)]
        return torch.concat(outputs,dim=1)





class myminRelu(nn.ReLU):
    def __init__(self, vmin):
        super(myminRelu, self).__init__()
        self.vmin = vmin

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input - self.vmin, inplace=self.inplace) + self.vmin


def get_activation(activation):
    if activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU(inplace=False)
    elif activation == "softplus":
        return nn.Softplus()
    elif activation == "linear":
        return nn.Identity()


class UNet(nn.Module):
    def __init__(self,input_channel, ngc,init_feature_kernel, paths,
                 output_channel,nb_ed_layers,generator_activation,
                 use_dropout,leaky_relu, norm, residual_layer=-1, blocks=("downconv-relu-norm", "convT-relu-norm"), ResUnet=False,
                 AttentionUnet=False,
                 dim=2,final_2dconv=False, final_2dchannels=0):
        super(UNet, self).__init__()

        self.ResUnet = ResUnet
        self.AttentionUnet = AttentionUnet
        self.input_channels = input_channel
        self.output_channels = output_channel

        if dim==2:
            self.dim = 2
            conv = nn.Conv2d
            init_feature_kernel_size,init_feature_stride,init_feature_padding = (int(init_feature_kernel), int(init_feature_kernel)),(1,1), int(init_feature_kernel / 2)
            final_kernel,final_stride,final_padding = (3,3), (1,1), 1
        elif dim==3:
            self.dim=3
            conv = nn.Conv3d
            init_feature_kernel_size,init_feature_stride,init_feature_padding = (int(init_feature_kernel), int(init_feature_kernel), int(init_feature_kernel)),(1,1, 1), int(init_feature_kernel / 2)
            final_kernel,final_stride,final_padding = (3,3,3), (1,1,1), 1

        block_e,block_d = blocks[0], blocks[1]

        if paths:
            self.paths=True

            # self.inital_paths = nn.Sequential(*[networks_attention_cbam.CBAM(112, reduction = 8) for _ in range(self.input_channels)])
            nb_channels_per_paths = ngc // input_channel
            self.init_feature = PathsBlock(input_channels=input_channel,nb_channels_per_paths=nb_channels_per_paths,
                                           nconv=2,
                                           kernel_size=init_feature_kernel_size, stride=init_feature_stride,padding=init_feature_padding)
        else:
            self.paths = False
            self.init_feature = conv(input_channel, ngc, kernel_size=init_feature_kernel_size, stride=init_feature_stride, padding = init_feature_padding)

        self.nb_ed_layers = nb_ed_layers
        down_layers = []
        up_layers = []
        # Contracting layers :
        k = 1
        for el in range(self.nb_ed_layers+1):
            if el < nb_ed_layers:
                down_layers.append(DownSamplingBlock(k * ngc,2 * k * ngc, norm = norm,leaky_relu_val=leaky_relu, block=block_e, res_unit=self.ResUnet, dim=dim))
                k = 2 * k
            else:
                down_layers.append(DownSamplingBlock(k * ngc,2*k * ngc, norm = norm,leaky_relu_val=leaky_relu, block=block_e, res_unit=self.ResUnet, dim=dim, last=True))


        self.down_layers = nn.Sequential(*down_layers)

        self.pool = torch.nn.MaxPool3d((3,3,3),(2,2,2),(1,1,1))

        for dl in range(self.nb_ed_layers):
            up_layers.append(UpSamplingBlockBis(k*ngc*2,k*ngc,
                                                norm=norm, use_dropout=use_dropout,
                                                leaky_relu_val=leaky_relu, block=block_d,
                                                res_unit=self.ResUnet, dim=dim))
            k = k//2
        self.up_layers = nn.Sequential(*up_layers)
        self.final_feature = conv(2*ngc, output_channel, kernel_size=final_kernel, stride=final_stride, padding = final_padding)
        # self.final_feature = conv(2*ngc, output_channel, kernel_size=(1,1,1), stride=(1,1,1), padding = (0,0,0))

        # # Core layer
        # # If any dropout layer is used, it is here
        # up_layers.append(UpSamplingBlock(k * ngc, int(k/2) * ngc, norm=norm, use_dropout=use_dropout,leaky_relu_val=leaky_relu, block=block_d,res_unit=self.ResUnet, dim=dim))
        # # Extracting layers :
        # for _ in range(self.nb_ed_layers - 1):
        #     up_layers.append(UpSamplingBlock(k * ngc, int(k / 4) * ngc, norm = norm,leaky_relu_val=leaky_relu, block=block_d, res_unit=self.ResUnet, dim=dim))
        #     k = int( k / 2)
        # self.up_layers = nn.Sequential(*up_layers)
        # self.final_feature = conv(2 * ngc, output_channel, kernel_size=final_kernel, stride=final_stride, padding = final_padding)

        self.residual_layer=residual_layer

        self.activation = get_activation(generator_activation)

        if final_2dconv:
            self.do_final_2d_conv=True
            self.final_2d_conv=nn.Conv2d(in_channels=final_2dchannels, out_channels=1, stride=(1,1), kernel_size=(3,3),padding=1)
        else:
            self.do_final_2d_conv=False

        if self.AttentionUnet:
            config = get_3DReg_config()
            patch_size =_triple(config.patches["size"])
            self.patch_embeddings = nn.Conv3d(in_channels=ngc*(2**self.nb_ed_layers),
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
            S = config.patches["size"][0]
            self.patch_de_embeddings = nn.ConvTranspose3d(in_channels=config.hidden_size,
                                                                  out_channels=ngc*(2**self.nb_ed_layers),
                                                                  kernel_size=(S,S,S),
                                                                  stride=(S,S,S))
            self.VisTrans = VisionTransformer(config=config,vis=False)

    def forward(self, x):
        if self.residual_layer>=0:
            if self.dim==2:
                residual=x[:,0:self.output_channels,:,:] if self.input_channels != self.output_channels else x
            elif self.dim==3:
                residual = x[:,self.residual_layer:self.residual_layer+1,:,:,:]


        # if self.paths:
        #     # different inital paths
        #     x_=[]
        #     for c in range(x.shape[1]):
        #         # x_.append(self.inital_paths[c](x[:,c:c+1,:,:,:]))
        #         x_.append(self.inital_paths[c](x[:,c,:,:,:])[:,None,:,:,:])
        #     x = torch.concatenate(x_, dim=1)


        # ----------------------------------------------------------
        #first feature extraction
        x0 = self.init_feature(x) # nhc
        # ----------------------------------------------------------
        # Contracting layers :
        list_xk  = [x0]
        xk = x0
        for l in range(self.nb_ed_layers+1):
            xk = self.down_layers[l](xk)
            list_xk.append(xk)
            if l<self.nb_ed_layers:
                xk = self.pool(xk)
        # ----------------------------------------------------------

        if self.AttentionUnet:
            B = xy.shape[0]
            p = self.patch_embeddings(xy)
            NP = p.shape[-1]
            p = p.flatten(2)
            p = p.transpose(-1,-2)
            p,_ = self.VisTrans(p)
            p = p.permute(0,2,1)
            H = p.shape[1]
            p = p.contiguous().view(B,H,NP,NP,NP)
            xy = self.patch_de_embeddings(p)
        # ----------------------------------------------------------
        # Extracting layers :

        # for l in range(self.nb_ed_layers):
        #     y = self.up_layers[l](xy)
        #     xy = torch.cat([xk[-l-2],y],1)

        for l in range(self.nb_ed_layers):
            y = self.up_layers[l](xk,list_xk[-2-l])
            xk = y
        # ----------------------------------------------------------
        # Final feature extraction
        y = self.final_feature(xk) # output_channel

        # # residual
        if self.residual_layer>=0:
            y += residual

        if self.do_final_2d_conv:
            y = y[:,0,:,:,:]
            y = self.final_2d_conv(y)[:,None,:,:,:]

        y = self.activation(y)
        # ----------------------------------------------------------
        return(y)

class CNN_block(nn.Module):
    def __init__(self, ic, oc,k,norm,leaky_relu_val=0.2, block="conv-norm-relu", res_unit=False):
        super(CNN_block, self).__init__()
        sequence = []
        splited_block = block.split('-')

        self.dim = 3
        conv = nn.Conv3d
        kernel_size = (k,k,k)
        stride = (1,1,1)
        padding = (int(k-1)//2, int(k-1)//2, int(k-1)//2)


        if norm == "batch_norm":
            norm_layer = nn.BatchNorm3d
        elif norm == "inst_norm":
            norm_layer = nn.InstanceNorm3d
        else:
            norm_layer = nn.Identity

        self.res_unit = res_unit


        for elmt in splited_block:
            if (elmt == 'conv'):
                sequence.append(
                    conv(ic, oc, kernel_size=kernel_size, stride=stride, padding=padding))
            elif elmt == "relu":
                sequence.append(nn.LeakyReLU(leaky_relu_val, inplace=False))
            elif elmt == "norm":
                sequence.append(norm_layer(oc))

        self.sequenceBlock = nn.Sequential(*sequence)

    def forward(self, x):
        if self.res_unit:
            return (x + self.sequenceBlock(x))
        else:
            return self.sequenceBlock(x)


class CNN(nn.Module):
    def __init__(self, input_channel, ngc, kernel, paths,
                 output_channel, nb_ed_layers, generator_activation,
                 leaky_relu, norm, residual_layer=-1, blocks=("conv-norm-relu"),
                 ResUnet=False):
        super(CNN, self).__init__()

        self.ResUnet = ResUnet
        self.input_channels = input_channel
        self.output_channels = output_channel
        block = blocks[0]
        self.nb_ed_layers = nb_ed_layers

        if paths==False:
            self.paths = False
            CNN_sequence = []
            CNN_sequence.append(CNN_block(
                ic=input_channel, oc=ngc, k=kernel, norm=norm, leaky_relu_val=leaky_relu,
                block=block, res_unit=False,
            ))
            for _ in range(self.nb_ed_layers-2):
                CNN_sequence.append(CNN_block(
                    ic=ngc,oc=ngc,k = kernel, norm=norm, leaky_relu_val=leaky_relu,
                    block=block,res_unit=ResUnet,
                ))
            CNN_sequence.append(CNN_block(
                ic=ngc, oc=output_channel, k=kernel, norm=norm, leaky_relu_val=leaky_relu,
                block=block, res_unit=False,
            ))
            self.CNN_sequence = nn.Sequential(*CNN_sequence)
        else:
            self.paths = True
            indiv_paths = []
            for _ in range(self.input_channels):
                indiv_path = []
                indiv_path.append(CNN_block(
                    ic=1, oc=ngc//2, k=kernel, norm=norm, leaky_relu_val=leaky_relu,
                    block=block, res_unit=False))
                for __ in range(self.nb_ed_layers//2):
                    indiv_path.append(CNN_block(
                        ic=ngc//2, oc=ngc//2, k=kernel, norm=norm, leaky_relu_val=leaky_relu,
                        block=block, res_unit=ResUnet))
                indiv_paths.append(nn.Sequential(*indiv_path))

            self.indiv_paths = nn.Sequential(*indiv_paths)

            common_path = []
            for _ in range(self.nb_ed_layers // 2):
                common_path.append(CNN_block(
                    ic=ngc, oc=ngc, k=kernel, norm=norm, leaky_relu_val=leaky_relu,
                    block=block, res_unit=ResUnet))

            common_path.append(CNN_block(
                ic=ngc, oc=output_channel, k=kernel, norm=norm, leaky_relu_val=leaky_relu,
                block=block, res_unit=False,
            ))
            self.common_path = nn.Sequential(*common_path)
            print("oooo"*20)
            print(self.indiv_paths)
            print('oooo'*20)
        self.residual_layer= residual_layer
        self.activation = get_activation(generator_activation)

    def forward(self, x):
        if self.residual_layer >= 0:
            residual = x[:, self.residual_layer:self.residual_layer + 1, :, :, :]

        # ----------------------------------------------------------
        if self.paths==False:
            y = self.CNN_sequence(x)
        else:
            paths_output = torch.concat(tuple([self.indiv_paths[k](x[:,k:k+1,:,:,:]) for k in range(self.input_channels)]),dim=1)
            y = self.common_path(paths_output)
        # ----------------------------------------------------------
        if self.residual_layer >= 0:
            y += residual

        y = self.activation(y)
        return (y)


class UNET_3D_2D(nn.Module):
    def __init__(self,input_channel, residual_layer=False,final_2dchannels=0):
        super(UNET_3D_2D, self).__init__()

        self.final_2dchannels = final_2dchannels
        self.residual_layer = residual_layer
        list_3d_channels=[input_channel, 16, 32, 64, 64,128, 1]

        sequence_3D = []

        for c in range(len(list_3d_channels)-1):
            sequence_3D.append(nn.Conv3d(in_channels=list_3d_channels[c],
                                         out_channels=list_3d_channels[c+1],
                                         kernel_size=(3,3,3),
                                         stride=(1,1,1),
                                         padding = 1))
            sequence_3D.append(nn.InstanceNorm3d(list_3d_channels[c + 1]))
            sequence_3D.append(nn.ReLU())

        self.sequence_3D = nn.Sequential(*sequence_3D)

        sequence_2D = []

        for c in range(4):
            sequence_2D.append(nn.Conv2d(in_channels=final_2dchannels,
                                         out_channels=final_2dchannels,
                                         kernel_size=(3,3),
                                         stride=(1,1),
                                         padding = 1))
            sequence_2D.append(nn.InstanceNorm2d(final_2dchannels))
            sequence_2D.append(nn.ReLU())

        sequence_2D.append(nn.Conv2d(in_channels=final_2dchannels,
                                     out_channels=1,
                                     kernel_size=(3, 3),
                                     stride=(1, 1),
                                     padding=1))
        sequence_2D.append(nn.InstanceNorm2d(final_2dchannels))
        sequence_2D.append(nn.ReLU())


        self.sequence_2D = nn.Sequential(*sequence_2D)


    def forward(self, x):
        if self.residual_layer:
            residual = x[:, 1:2,self.final_2dchannels//2,:,:]

        y = self.sequence_3D(x)
        y = self.sequence_2D(y[:,0,:,:,:])

        if self.residual_layer:
            return (y+residual)[:,None,:,:,:]
        else:
            return y[:,None,:,:,:]





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


# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


# Define the Deconvolution Network
class ResCNN(nn.Module):
    def __init__(self, in_channels, out_channels, ngc=64):
        super(ResCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, ngc, kernel_size=(7,7), stride=(1,1), padding=3)
        self.relu = nn.ReLU(inplace=True)

        self.residual_blocks = nn.Sequential(
            ResidualBlock(ngc, ngc),
            ResidualBlock(ngc, ngc),
            ResidualBlock(ngc, ngc)
        )

        self.final_layer = nn.Conv2d(ngc, out_channels, kernel_size=(1,1), stride=(1,1), padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.residual_blocks(x)

        x = self.final_layer(x)
        return x


class vanillaCNN(nn.Module):
    def __init__(self,input_channel, ngc,init_feature_kernel,
                 output_channel,nb_ed_layers,generator_activation,
                 use_dropout,leaky_relu, norm, residual_layer=False,
                 dim=2):
        super(vanillaCNN, self).__init__()

        self.residual_layer= residual_layer
        self.input_channels = input_channel
        self.output_channels = output_channel

        if dim==2:
            self.dim = 2
            conv = nn.Conv2d
            init_feature_kernel_size,init_feature_stride,init_feature_padding = (int(init_feature_kernel), int(init_feature_kernel)),(1,1), int(init_feature_kernel / 2)
            conv_kernels,conv_strides,conv_paddings = (3,3), (1,1), 1

            if norm=="batch_norm":
                norm_layer = nn.BatchNorm2d
            elif norm=="inst_norm":
                norm_layer = nn.InstanceNorm2d
            else:
                norm_layer = nn.Identity

        elif dim==3:
            self.dim=3
            conv = nn.Conv3d
            init_feature_kernel_size,init_feature_stride,init_feature_padding = (int(init_feature_kernel), int(init_feature_kernel), int(init_feature_kernel)),(1,1,1), int(init_feature_kernel / 2)
            # conv_kernels,conv_strides,conv_paddings = (3,3,3), (1,1,1), 1
            conv_kernels,conv_strides,conv_paddings = (7,7,7), (1,1,1), 3

            if norm=="batch_norm":
                norm_layer = nn.BatchNorm3d
            elif norm=="inst_norm":
                norm_layer = nn.InstanceNorm3d
            else:
                norm_layer = nn.Identity

        sequence = []

        # First Layer
        sequence.append(conv(input_channel,
                                      ngc,
                                      kernel_size=init_feature_kernel_size,
                                      stride=init_feature_stride,
                                      padding = init_feature_padding))
        sequence.append(nn.ReLU())
        sequence.append(norm_layer(ngc))

        if use_dropout:
            sequence.append(nn.Dropout(0.3))

        # Inner layers
        for k in range(nb_ed_layers - 2):
            sequence.append(conv(ngc,
                                  ngc,
                                  kernel_size=conv_kernels,
                                  stride=conv_strides,
                                  padding = conv_paddings))
            sequence.append(nn.ReLU())
            sequence.append(norm_layer(ngc))

        # Last Layer
        sequence.append(conv(ngc,
                                  output_channel,
                                  kernel_size=conv_kernels,
                                  stride=conv_strides,
                                  padding = conv_paddings))


        # sequence.append(get_activation(generator_activation))

        self.sequence_CNN = nn.Sequential(*sequence)
        self.final_activation=get_activation(generator_activation)

    def forward(self,x):
        if self.residual_layer:
            if self.dim==2:
                residual=x[:,0:self.output_channels,:,:] if self.input_channels != self.output_channels else x
            elif self.dim==3:
                residual = x[:, 1:2,:,:,:]

            return self.final_activation(residual+self.sequence_CNN(x))
        else:
            return self.final_activation(self.sequence_CNN(x))


class Big3DUnet(nn.Module):
    def __init__(self, params, input_channels):
        super(Big3DUnet, self).__init__()

        norm = params["layer_norm"]
        self.use_dropout = params['use_dropout']
        self.residual_channel = 1 if params['residual_layer'] else -1

        # Encoder
        self.encoder1 = self.conv_block(input_channels, 32, norm = norm)
        self.encoder2 = self.conv_block(32, 64, norm = norm)
        self.encoder3 = self.conv_block(64, 128, norm = norm)
        self.encoder4 = self.conv_block(128, 256, norm = norm)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512, norm = norm)

        # Decoder
        self.upconv4 = self.upconv_block(512, 256)
        self.decoder4 = self.conv_block(512, 256, norm = norm)
        self.upconv3 = self.upconv_block(256, 128)
        self.decoder3 = self.conv_block(256, 128, norm = norm)
        self.upconv2 = self.upconv_block(128, 64)
        self.decoder2 = self.conv_block(128, 64, norm = norm)
        self.upconv1 = self.upconv_block(64, 32)
        self.decoder1 = self.conv_block(64, 32, norm = norm)

        if self.use_dropout:
            self.dropout = nn.Dropout3d(p=0.5)

        # Output
        self.output = nn.Conv3d(32, 1, kernel_size=1, padding='same')
        self.activation = nn.ReLU(inplace=True)

    def conv_block(self, in_channels, out_channels, norm):
        if norm=="batch_norm":
            norm=nn.BatchNorm3d(out_channels)
        elif norm=="inst_norm":
            norm=nn.InstanceNorm3d(out_channels)
        else:
            norm = nn.Identity(out_channels)

        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            norm,
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            norm,
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=2, padding='same')
        )

    def forward(self, x):
        res = x[:,self.residual_channel:self.residual_channel+1,:,:,:] if self.residual_channel>=0 else None

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool3d(e1, 2))
        e3 = self.encoder3(F.max_pool3d(e2, 2))
        e4 = self.encoder4(F.max_pool3d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool3d(e4, 2))

        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)
        if self.use_dropout:
            d4 = self.dropout(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        d0 = self.output(d1)


        # Output
        out = self.activation(d0+res) if self.residual_channel>=0 else self.activation(d0)
        return out
