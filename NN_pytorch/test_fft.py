# %%
import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms
from fastmri.models import Unet

import numpy as np
import pathlib
import torch.optim as optim
from fastmri.data import  mri_data
import math
import matplotlib.pyplot as plt
from my_data import *

# %% data loader
snr = 10
reso = 5

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

class Sample(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.weight = factor*N1/(N1-32*reso)*N2/(N2-32*reso) * torch.ones(N2-32*reso)
        self.sigma = sigma

    def forward(self,kspace):

        mask =  torch.zeros((N1,N2))
        mask[(16*reso):(N1-16*reso),(16*reso):(N2-16*reso)] =  1.0 / (self.weight ** 0.5).unsqueeze(0).repeat(N1-32*reso,1)
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(4).repeat(kspace.size(0),Nc,1,1,2)
        
        noise = self.sigma * torch.randn_like(kspace)
        kspace_noise =  (mask > 0) * kspace + mask * noise 

        return kspace_noise
    
class Recon(torch.nn.Module): 
    
    def __init__(self):
        super().__init__()
        self.weight =  0.01 * torch.ones(N1-32*reso, N2-32*reso)

    def forward(self,kspace):
        mask =  torch.zeros((N1,N2))
        mask[(16*reso):(N1-16*reso),(16*reso):(N2-16*reso)] =  self.weight
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(4).repeat(kspace.size(0),Nc,1,1,2)
        kspace =  mask * kspace 

        return kspace

def toIm(kspace,maps): 
    # kspace-(batch,Nc,N1,N2,2) maps-(batch,Nc,N1,N2,2)
    # image-(batch,N1,N2)
    kmask = torch.zeros_like(kspace)
    kmask[:,:,(16*reso):(N1-16*reso),(16*reso):(N2-16*reso),:] = 1
    image = fastmri.complex_abs(torch.sum(fastmri.complex_mul(fastmri.ifft2c(kmask*kspace),fastmri.complex_conj(maps)),dim=1))
    return image.squeeze()

# %% sampling
factor = 8
sigma =  0.12*math.sqrt(8)/snr

weight1 = torch.load('/home/wjy/Project/optsamp_model/opt_mask_window_snr'+str(snr)+'_reso'+str(reso))
sample_model = Sample(sigma,factor)
sample_model.weight = weight1

weight2 = torch.load('/home/wjy/Project/optsamp_model/opt_window_snr'+str(snr)+'_reso'+str(reso))
recon_model = Recon()
recon_model.weight = weight2

# %% data loader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_dataloader = torch.utils.data.DataLoader(test_data,batch_size,shuffle=True)

# %% optimization parameters
Loss = torch.nn.MSELoss()

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
ssim_module = SSIM(data_range=255, size_average=True, channel=1)

# %% test
count = 0
ssim_, nrmse_ = 0, 0

with torch.no_grad():
    for kspace, maps in test_dataloader:
    
        count += 1

        gt = toIm(kspace, maps) # ground truth
        scale = gt.max()
        l2scale = gt.norm(p=2)

        kspace_noise = recon_model(sample_model(kspace)) # add noise and apply window
        recon = toIm(kspace_noise, maps)
    
        ssim_ += ssim_module(recon.unsqueeze(0).unsqueeze(1)/scale*256, gt.unsqueeze(0).unsqueeze(1)/scale*256)
        nrmse_ += (recon-gt).norm(p=2)/l2scale

print('ssim: ', ssim_/count, ' nrmse: ', nrmse_/count)

# %%
