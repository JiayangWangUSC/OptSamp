# %%
import math
from typing import List, Tuple, Optional
import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms
from fastmri.models import Unet
import pathlib
import torch.optim as optim
from fastmri.data import  mri_data
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torchvision.transforms
from PIL import Image
import numpy as np
from my_data import *

# %% parameters
factor = 8
snr = 10
reso = 2
sigma =  0.12*math.sqrt(8)/snr

# %% data loader
N1 = 320
N2 = 320
Nc = 16

def data_transform(kspace,maps):
    # Transform the kspace to tensor format
    kspace = transforms.to_tensor(kspace)
    maps = transforms.to_tensor(maps)
    kspace = torch.cat((kspace[torch.arange(Nc),:,:].unsqueeze(3),kspace[torch.arange(Nc,2*Nc),:,:].unsqueeze(3)),3)
    maps = torch.cat((maps[torch.arange(Nc),:,:].unsqueeze(3),maps[torch.arange(Nc,2*Nc),:,:].unsqueeze(3)),3)
    kspace = kspace.permute([0,2,1,3])
    maps = maps.permute([0,2,1,3]) 
    return kspace, maps

test_data = SliceDataset(
    root=pathlib.Path('/home/wjy/Project/fastmri_dataset/brain_T1_demo/'),
    #root = pathlib.Path('/project/jhaldar_118/jiayangw/dataset/brain_T1/multicoil_train/'),
    transform=data_transform,
    challenge='multicoil'
)

# %% noise generator and transform to image
batch_size = 1

class Sample_uni(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.weight =  torch.zeros((N1,N2))
        self.weight[(16*reso):(N1-16*reso),(16*reso):(N2-16*reso)] = 1/(factor*N1/(N1-32*reso)*N2/(N2-32*reso))
        self.sigma = sigma

    def forward(self,kspace):
        noise = self.sigma*torch.randn_like(kspace)
        mask = torch.sqrt(self.weight).unsqueeze(0).unsqueeze(0).unsqueeze(4).repeat(kspace.size(0),Nc,1,1,2) 
        kspace_noise = (mask>0) * kspace + noise * mask
        return kspace_noise
    
class Sample_opt(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.weight = factor*N1/(N1-32*reso)*N2/(N2-32*reso) * torch.ones(N2-32*reso)
        self.sigma = sigma

    def forward(self,kspace):
        mask =  torch.zeros((N1,N2))
        mask[(16*reso):(N1-16*reso),(16*reso):(N2-16*reso)] =  1.0 / (self.weight ** 0.5).unsqueeze(0).repeat(N1-32*reso,1)
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(4).repeat(kspace.size(0),Nc,1,1,2)
        noise = self.sigma*torch.randn_like(kspace)
        kspace_noise =  (mask>0) * kspace + mask * noise 
        return kspace_noise


def toIm(kspace,maps): 
    # kspace-(batch,Nc,N1,N2,2) maps-(batch,Nc,N1,N2,2)
    # image-(batch,N1,N2)
    kmask = torch.zeros_like(kspace)
    kmask[:,:,(16*reso):(N1-16*reso),(16*reso):(N2-16*reso),:] = 1
    image = fastmri.complex_abs(torch.sum(fastmri.complex_mul(fastmri.ifft2c(kmask*kspace),fastmri.complex_conj(maps)),dim=1))
    return image.squeeze()

def fullIm(kspace,maps): 
    # kspace-(batch,Nc,N1,N2,2) maps-(batch,Nc,N1,N2,2)
    # image-(batch,N1,N2)
    image = fastmri.complex_abs(torch.sum(fastmri.complex_mul(fastmri.ifft2c(kspace),fastmri.complex_conj(maps)),dim=1))
    return image.squeeze()

# %% GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_dataloader = torch.utils.data.DataLoader(test_data,batch_size,shuffle=True)

# %%
sample_uni = Sample_uni(sigma,factor)
recon_uni = torch.load('/home/wjy/Project/optsamp_model/uni_mse_snr'+str(snr)+'_reso'+str(reso),map_location=torch.device('cpu'))

weight = torch.load('/home/wjy/Project/optsamp_model/opt_mse_mask_snr'+str(snr)+'_reso'+str(reso))
sample_opt = Sample_opt(sigma,factor)
sample_opt.weight = weight
recon_opt = torch.load('/home/wjy/Project/optsamp_model/opt_mse_snr'+str(snr)+'_reso'+str(reso),map_location=torch.device('cpu'))

# %%
with torch.no_grad():
    kspace, maps =  test_data[0]
    kspace = kspace.unsqueeze(0)
    maps = maps.unsqueeze(0)
    gt = fullIm(kspace, maps)
    support = (gt > 0.03 * gt.max())
#    support = fastmri.complex_abs(torch.sum(fastmri.complex_mul(maps,fastmri.complex_conj(maps)),dim=1)).squeeze()

    # %% uni recon
    kspace_uni = sample_uni(kspace)
    noise_uni = fastmri.ifft2c(kspace_uni)
    input_uni = torch.cat((noise_uni[:,:,:,:,0],noise_uni[:,:,:,:,1]),1).to(device)
    output_uni = recon_uni(input_uni).to(device)
    image_uni = fastmri.complex_abs(torch.cat((output_uni[:,0,:,:].unsqueeze(1).unsqueeze(4),output_uni[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)    
    save_image(image_uni/gt.max()*1.5,'/home/wjy/Project/optsamp_result/uni_unet_snr'+str(snr)+'.png')
    save_image(support*(image_uni-gt).abs()/gt.max()*5,'/home/wjy/Project/optsamp_result/uni_error_snr'+str(snr)+'.png')

    #%% opt recon
    kspace_opt = sample_opt(kspace)
    noise_opt = fastmri.ifft2c(kspace_opt)
    input_opt = torch.cat((noise_opt[:,:,:,:,0],noise_opt[:,:,:,:,1]),1).to(device)
    output_opt = recon_opt(input_opt).to(device)
    image_opt = fastmri.complex_abs(torch.cat((output_opt[:,0,:,:].unsqueeze(1).unsqueeze(4),output_opt[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
    save_image((0.95*image_opt+0.05*gt)/gt.max()*1.5,'/home/wjy/Project/optsamp_result/opt_unet_snr'+str(snr)+'.png')
    save_image((0.95*image_opt+0.05*gt-gt).abs()/gt.max()*5,'/home/wjy/Project/optsamp_result/opt_error_snr'+str(snr)+'.png')

# %% save error map
error = (image_opt-gt).abs()/gt.max()
error = error.squeeze().numpy()
plt.imshow(error, cmap='hot',vmax=0.2,vmin=0)
plt.axis('off')
plt.savefig('/home/wjy/Project/optsamp_result/opt_error_snr'+str(snr)+'.png', bbox_inches='tight', pad_inches=0) 

# %% save patch
resize_transform = torchvision.transforms.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

patch1 = resize_transform(gt[160:240,100:180].unsqueeze(0))/torch.max(gt)*1.5
save_image(patch1,'/home/wjy/Project/optsamp_result/gt_p1_snr'+str(snr)+'.png')
