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
from PIL import Image
import numpy as np

from my_data import *

# %% data loader
N1 = 320
N2 = 320
Nc = 20
def data_transform(kspace,maps):
    # Transform the kspace to tensor format
    kspace = transforms.to_tensor(kspace)
    maps = transforms.to_tensor(maps)
    kspace = torch.cat((kspace[torch.arange(Nc),:,:].unsqueeze(3),kspace[torch.arange(Nc,2*Nc),:,:].unsqueeze(3)),3)
    maps = torch.cat((maps[torch.arange(Nc),:,:].unsqueeze(3),maps[torch.arange(Nc,2*Nc),:,:].unsqueeze(3)),3)
    kspace = kspace.permute([0,2,1,3])
    maps = maps.permute([0,2,1,3]) + 1e-7

    return kspace, maps

test_data = SliceDataset(
    root=pathlib.Path('/home/wjy/Project/fastmri_dataset/brain_T1_demo/'),
    #root = pathlib.Path('/project/jhaldar_118/jiayangw/dataset/brain_T1/multicoil_train/'),
    transform=data_transform,
    challenge='multicoil'
)

#val_data = mri_data.SliceDataset(
#    #root=pathlib.Path('/home/wjy/Project/fastmri_dataset/multicoil_test/T2/'),
#    root = pathlib.Path('/project/jhaldar_118/jiayangw/OptSamp/dataset/val/'),
#    transform=data_transform,
#    challenge='multicoil'
#)

# %% noise generator and transform to image
batch_size = 8

class Sample_opt(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.weight = factor*torch.ones(N2)
        self.factor = factor
        self.sigma = sigma

    def forward(self,kspace):
        #sample_mask = torch.sqrt(1 + F.softmax(self.mask)*(self.factor-1)*N2)
        
        support = self.weight >= 1
        support = support.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(4).repeat(kspace.size(0),Nc,N1,1,2)

        mask = self.weight.clone() 
        mask = 1.0 / (self.weight ** 0.5)
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(4).repeat(kspace.size(0),Nc,N1,1,2)
        
        
        noise = self.sigma*torch.randn_like(kspace)
        kspace_noise = support * (kspace + mask * noise) 
        return kspace_noise

class Sample_low50(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.mask = factor*torch.ones(N2)
        self.sigma = sigma

    def forward(self,kspace):
        noise = self.sigma*torch.randn_like(kspace)
        
        # low_50(80,240)
        # low_25(120,200)
        support = torch.zeros(N2)
        support[torch.arange(80,240)] = 1
        noise = noise/math.sqrt(factor*2)
        
        kspace_noise = torch.mul(kspace + noise, support.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(0).repeat(kspace.size(0),Nc,N1,1,2))
        
        return kspace_noise

class Sample_low25(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.mask = factor*torch.ones(N2)
        self.sigma = sigma

    def forward(self,kspace):
        noise = self.sigma*torch.randn_like(kspace)
        
        # low_50(80,240)
        # low_25(120,200)
        support = torch.zeros(N2)
        support[torch.arange(120,200)] = 1
        noise = noise/math.sqrt(factor*4)
        
        kspace_noise = torch.mul(kspace + noise, support.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(0).repeat(kspace.size(0),Nc,N1,1,2))
        
        return kspace_noise

class Sample_uni(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.mask = factor*torch.ones(N2)
        self.sigma = sigma

    def forward(self,kspace):
        noise = self.sigma*torch.randn_like(kspace)
        kspace_noise = kspace + torch.div(noise,torch.sqrt(self.mask).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(4).repeat(kspace.size(0),Nc,N1,1,2))  # need to reshape mask        image = fastmri.ifft2c(kspace_noise)
        return kspace_noise


def toIm(kspace,maps): 
    # kspace-(batch,Nc,N1,N2,2) maps-(batch,Nc,N1,N2,2)
    # image-(batch,N1,N2)
    image = fastmri.complex_abs(torch.sum(fastmri.complex_mul(fastmri.ifft2c(kspace),fastmri.complex_conj(maps)),dim=1))
    return image.squeeze()

# %% parameters
factor = 8
snr = 10
sigma =  math.sqrt(8)*45/snr

# %% GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_dataloader = torch.utils.data.DataLoader(test_data,batch_size,shuffle=True)

# %%
weight = torch.load('/home/wjy/Project/optsamp_model/opt_mse_mask_snr'+str(snr))

sample_uni = Sample_uni(sigma,factor)
sample_low50 = Sample_low50(sigma,factor)
sample_low25 = Sample_low25(sigma,factor)
sample_opt = Sample_opt(sigma,factor)
sample_opt.weight = weight


recon_uni = torch.load('/home/wjy/Project/optsamp_model/uni_mse_snr'+str(snr),map_location=torch.device('cpu'))
recon_low50 = torch.load('/home/wjy/Project/optsamp_model/low50_mse_snr'+str(snr),map_location=torch.device('cpu'))
recon_low25 = torch.load('/home/wjy/Project/optsamp_model/low25_mse_snr'+str(snr),map_location=torch.device('cpu'))
recon_opt = torch.load('/home/wjy/Project/optsamp_model/opt_mse_snr'+str(snr),map_location=torch.device('cpu'))


#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
#ssim_module = SSIM(data_range=255, size_average=True, channel=1)

#ssim_uni, ssim_opt, ssim_low = 0, 0, 0
#mse_uni, mse_opt, mse_low = 0, 0, 0 


# %% recon
with torch.no_grad():
  for kspace, maps in test_dataloader:
    gt = toIm(kspace, maps)
   
    # uni recon
    kspace_noise = sample_uni(kspace)
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_uni(image_input)
    image_recon = torch.cat((image_output[:,torch.arange(Nc),:,:].unsqueeze(4),image_output[:,torch.arange(Nc,2*Nc),:,:].unsqueeze(4)),4)
    image_uni = fastmri.complex_abs(torch.sum(fastmri.complex_mul(image_recon,fastmri.complex_conj(maps.to(device))),dim=1)).squeeze()

    # lwo50 recon
    kspace_noise = sample_low50(kspace)
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_low50(image_input)
    image_recon = torch.cat((image_output[:,torch.arange(Nc),:,:].unsqueeze(4),image_output[:,torch.arange(Nc,2*Nc),:,:].unsqueeze(4)),4)
    image_low50 = fastmri.complex_abs(torch.sum(fastmri.complex_mul(image_recon,fastmri.complex_conj(maps.to(device))),dim=1)).squeeze()

    # lwo25 recon
    kspace_noise = sample_low25(kspace)
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_low25(image_input)
    image_recon = torch.cat((image_output[:,torch.arange(Nc),:,:].unsqueeze(4),image_output[:,torch.arange(Nc,2*Nc),:,:].unsqueeze(4)),4)
    image_low25 = fastmri.complex_abs(torch.sum(fastmri.complex_mul(image_recon,fastmri.complex_conj(maps.to(device))),dim=1)).squeeze()

    # opt recon
    kspace_noise = sample_opt(kspace)
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_opt(image_input)
    image_recon = torch.cat((image_output[:,torch.arange(Nc),:,:].unsqueeze(4),image_output[:,torch.arange(Nc,2*Nc),:,:].unsqueeze(4)),4)
    image_opt = fastmri.complex_abs(torch.sum(fastmri.complex_mul(image_recon,fastmri.complex_conj(maps.to(device))),dim=1)).squeeze()

# %%
