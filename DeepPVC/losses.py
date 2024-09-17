import torch
from torch import nn
from torch.cuda.amp import custom_fwd
from . import helpers

def get_nn_loss(loss_name):
    if loss_name=="L1":
        return nn.L1Loss()
    elif loss_name=="L2":
        return nn.MSELoss()
    elif loss_name=="Poisson":
        return PoissonLikelihood_loss()
    elif loss_name=='BCE':
        return nn.BCEWithLogitsLoss()
    elif loss_name=="Wasserstein":
        return Wasserstein_loss()
    elif loss_name=="Sum":
        return Sum_loss()
    elif loss_name=="SmoothL1":
        return nn.SmoothL1Loss()
    elif loss_name=="lesion":
        return nn.L1Loss()
    elif loss_name=="conv":
        return nn.L1Loss()
    elif loss_name=="edcc":
        return eDCC_loss()
    else:
        print(f'ERROR in loss name {loss_name}')
        exit(0)

class Lesion_loss(nn.Module):
    def __init__(self):
        super(Lesion_loss, self).__init__()
        self.l1=nn.L1Loss()

    @custom_fwd
    def forward(self, y_true, y_pred, lesion_mask):
        return self.l1(y_true[lesion_mask], y_pred[lesion_mask])

class PoissonLikelihood_loss(nn.Module):
    def __init__(self):
        super(PoissonLikelihood_loss, self).__init__()

    @custom_fwd
    def forward(self, y_true,y_pred):
        eps = 1e-6
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_true = y_true.view(y_true.shape[0], -1)

        p_l = -y_true+y_pred*torch.log(y_true + eps)-torch.lgamma(y_pred+1)
        return -torch.mean(p_l)

class eDCC_loss(nn.Module):
    def __init__(self):
        super(eDCC_loss, self).__init__()
        self.device = helpers.get_auto_device("auto")
        self.spacing = 4.7952
        self.mu0 = 0.013
        self.Nprojs = 128
        self.array_theta = torch.linspace(0, 2 * torch.pi, self.Nprojs + 1)[:-1]

    def laplace_p(self,p,sigma):
        size = p.shape[-1]
        exp_s = torch.tensor([torch.exp(sigma*s*self.spacing)
                                         for s in
                                         torch.linspace((-size*self.spacing+self.spacing)/2,
                                                        (size*self.spacing-self.spacing)/2,size)],
                                        device=self.device)
        return torch.matmul(p,exp_s)

    @custom_fwd
    def forward(self,_,projs):
        for i,thetai in enumerate(self.array_theta):
            j =(i+30)%128
            thetaj = self.array_theta[j]
            edcc = None
            for l in range(projs.shape[-2]):
                sigma_ij = self.mu0 * torch.tan(torch.tensor([(thetai-thetaj)/2]))
                sigma_ji = self.mu0 * torch.tan(torch.tensor([(thetai-thetaj)/2]))
                P_i = self.laplace_p(p=projs[:,i,l,:],sigma=sigma_ij)
                P_j = self.laplace_p(p=projs[:,j,l,:],sigma=sigma_ji)
                if edcc is None:
                    edcc = 2/self.Nprojs * torch.abs(P_i-P_j)/(P_i+P_j) if (P_i+P_j != 0) else 0
                else:
                    edcc += 2/self.Nprojs * torch.abs(P_i-P_j)/(P_i+P_j) if (P_i+P_j != 0) else 0

        return edcc.mean()

class gradient_penalty(nn.Module):
    # cf https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
    def __init__(self,device):
        super(gradient_penalty, self).__init__()
        self.device = device

    def switch_device(self,device):
        self.device=device

    @custom_fwd
    def forward(self,interpolates, model_interpolates):
        grad_outputs = torch.ones(model_interpolates.size(),device=self.device, requires_grad=False)

        gradients = torch.autograd.grad(
            outputs=model_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        return gradient_penalty


class Wasserstein_loss(nn.Module):
    def __init__(self):
        super(Wasserstein_loss, self).__init__()

    @custom_fwd
    def forward(self, D_disc,target):
        return (torch.mean( (-2*target+1) * D_disc))

class Sum_loss(nn.Module):
    def __init__(self):
        super(Sum_loss, self).__init__()

    @custom_fwd
    def forward(self,target,output):
        return torch.abs(target.sum() - output.sum())


class Model_Loss(nn.Module):
    def __init__(self, losses_params):
        super().__init__()
        self.loss_name, self.recon_loss,self.lambdas=[],[],[]
        if type(losses_params['recon_loss'])==list:
            for (loss,lbda) in zip(losses_params['recon_loss'],losses_params['lambda_recon']):
                self.recon_loss.append(get_nn_loss(loss_name=loss))
                self.lambdas.append(lbda)
                self.loss_name.append(loss)
        else:
            self.recon_loss=[get_nn_loss(loss_name=losses_params['recon_loss'])]
            self.lambdas=[1]
            self.loss_name = [losses_params['recon_loss']]

        self.device = losses_params['device']

    def extra_repr(self):
        extra_repr_str=''
        extra_repr_str+='(recon_loss):'
        for loss in self.recon_loss:
            mod_str = repr(loss)
            extra_repr_str+='  '+mod_str+','
        extra_repr_str+='\n'
        extra_repr_str+='(lambdas_recon):'
        for lbda in self.lambdas:
            extra_repr_str+='  ' + str(lbda)+','

        return extra_repr_str


class Pix2PixLosses(Model_Loss):
    def __init__(self, losses_params):
        super().__init__(losses_params=losses_params)
        self.adv_loss = get_nn_loss(loss_name=losses_params['adv_loss'])

        if losses_params['gradient_penalty']:
            self.gradient_penalty = gradient_penalty(device=self.device)

        self.ones = torch.tensor([0.0],device=self.device)

    def get_adv_loss(self):
        return self.adv_loss
    def get_recon_loss(self,target,output):
        weighted_recon_loss=sum([lbda * loss(target,output) for (lbda,loss) in zip(self.lambdas,self.recon_loss)])
        return weighted_recon_loss

    def get_gen_loss(self, disc_fake_hat, truePVfree, fakePVfree):
        # gen_adv_loss = self.adv_loss(disc_fake_hat, torch.ones_like(disc_fake_hat))
        gen_adv_loss = self.adv_loss(disc_fake_hat, self.ones.expand_as(disc_fake_hat))
        gen_rec_loss = self.get_recon_loss(truePVfree, fakePVfree)
        gen_loss = gen_adv_loss + gen_rec_loss
        return gen_loss

    def get_gradient_penalty(self,Discriminator,real,fake,condition):
        alpha = torch.randn((real.size(0), 1, 1, 1), device=self.device)
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
        model_interpolates = Discriminator(interpolates,condition)
        return self.gradient_penalty(interpolates, model_interpolates)


class UNetLosses(Model_Loss):
    def __init__(self, losses_params):
        super().__init__(losses_params)

    def get_unet_loss(self, target, output, lesion_mask=None, conv_psf=None, input_rec=None):
        unet_loss = 0
        for (loss_name,loss,lbda) in zip(self.loss_name,self.recon_loss,self.lambdas):
            if loss_name=="lesion":
                unet_loss+= lbda * loss(target[lesion_mask], output[lesion_mask])
            elif loss_name=="conv":
                unet_loss+= lbda * loss(input_rec[:,None,:,:,:], conv_psf(output[:,None,:,:,:]))
            else:
                unet_loss+= lbda * loss(target, output)
        return unet_loss