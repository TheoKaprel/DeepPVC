

from .Model_Pix2Pix import Pix2PixModel
from .Model_temp_others import GAN_Denoiser_PVC
from .Model_Unet import UNetModel
from .Model_Unet_Denoiser_PVC import UNet_Denoiser_PVC
from .Model_Diffusion import Diffusion_UNet
from .Model_Unet_Double_Domain import UNet_Double_Domain

class ModelInstance():
    def __new__(cls, params, from_pth = None, resume_training=False, device = None):
        network_architecture = params['network']

        if network_architecture == 'pix2pix':
            return Pix2PixModel(params=params, from_pth=from_pth,resume_training=resume_training, device=device)
        elif network_architecture == 'unet':
            return UNetModel(params=params, from_pth=from_pth,resume_training=resume_training, device=device)
        elif network_architecture == 'unet_denoiser_pvc':
            return UNet_Denoiser_PVC(params=params, from_pth=from_pth,resume_training=resume_training, device=device)
        elif network_architecture == 'double_domain':
            return UNet_Double_Domain(params=params, from_pth=from_pth,resume_training=resume_training, device=device)
        elif network_architecture == 'gan_denoiser_pvc':
            return GAN_Denoiser_PVC(params=params, from_pth=from_pth,resume_training=resume_training, device=device)
        elif network_architecture == 'diffusion':
            return Diffusion_UNet(params=params, from_pth = from_pth, resume_training = resume_training, device = device)
        else:
            print(f"ERROR : unknown network architecture ({network_architecture})")
            exit(0)
