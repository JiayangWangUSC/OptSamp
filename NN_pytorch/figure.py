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
snr = 3
reso = 0
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
    root=pathlib.Path('/home/wjy/Project/fastmri_dataset/brain_T1/'),
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


# %% single image recon
seed = 360
with torch.no_grad():
    kspace, maps = test_data[0]  
    kspace = kspace.unsqueeze(0)
    maps = maps.unsqueeze(0)
    gt = toIm(kspace, maps).squeeze()
    support = fastmri.complex_abs(torch.sum(fastmri.complex_mul(maps,fastmri.complex_conj(maps)),dim=1))
   
    # uni100 recon
    torch.manual_seed(seed=seed)
    kspace_noise = sample_uni100(kspace)
    image_noise_uni100 = toIm(kspace_noise, maps).squeeze()
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_uni100(image_input)
    recon = fastmri.complex_abs(torch.cat((image_output[:,0,:,:].unsqueeze(1).unsqueeze(4),image_output[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
    image_uni100 = recon * support.to(device)

    # opt100 recon
    torch.manual_seed(seed=seed)
    kspace_noise = sample_opt(kspace)
    image_noise_opt = toIm(kspace_noise, maps).squeeze()
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_opt(image_input)
    recon = fastmri.complex_abs(torch.cat((image_output[:,0,:,:].unsqueeze(1).unsqueeze(4),image_output[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
    image_opt = recon * support.to(device)

# %% save single image
save_image(gt/gt.max()*1.5,'/home/wjy/Project/optsamp_result/gt.png')
save_image(image_uni100/gt.max()*1.5,'/home/wjy/Project/optsamp_result/uni100_snr'+str(snr)+'.png')
save_image(image_opt/gt.max()*1.5,'/home/wjy/Project/optsamp_result/opt_snr'+str(snr)+'.png')

# %% save error map
error = (image_uni100-gt).abs()/gt.max()
error = error.squeeze().numpy()
plt.imshow(error, cmap='hot',vmax=0.14,vmin=0.028)
plt.axis('off')
plt.savefig('/home/wjy/Project/optsamp_result/uni100_error_snr'+str(snr)+'.png', bbox_inches='tight', pad_inches=0) 

# %% save patch
resize_transform = torchvision.transforms.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

patch1 = resize_transform(gt[160:240,100:180].unsqueeze(0))/torch.max(gt)*1.5
save_image(patch1,'/home/wjy/Project/optsamp_result/gt_p1_snr'+str(snr)+'.png')
