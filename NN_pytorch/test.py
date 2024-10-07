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
    maps = maps.permute([0,2,1,3]) + 1e-7

    return kspace, maps

test_data = SliceDataset(
    root=pathlib.Path('/home/wjy/Project/fastmri_dataset/brain_T1/test/'),
    #root = pathlib.Path('/project/jhaldar_118/jiayangw/dataset/brain_T1/multicoil_train/'),
    transform=data_transform,
    challenge='multicoil'
)

# %% noise generator and transform to image
batch_size = 1

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
snr = 3
sigma =  math.sqrt(8)*45/snr

# %% GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_dataloader = torch.utils.data.DataLoader(test_data,batch_size,shuffle=True)

# %%
weight = torch.load('/home/wjy/Project/optsamp_model/opt_mae_mask_snr'+str(snr))

sample_uni = Sample_uni(sigma,factor)
sample_low50 = Sample_low50(sigma,factor)
sample_low25 = Sample_low25(sigma,factor)
sample_opt = Sample_opt(sigma,factor)
sample_opt.weight = weight

recon_uni = torch.load('/home/wjy/Project/optsamp_model/uni_mae_snr'+str(snr),map_location=torch.device('cpu'))
recon_low50 = torch.load('/home/wjy/Project/optsamp_model/low50_mae_snr'+str(snr),map_location=torch.device('cpu'))
recon_low25 = torch.load('/home/wjy/Project/optsamp_model/low25_mae_snr'+str(snr),map_location=torch.device('cpu'))
recon_opt = torch.load('/home/wjy/Project/optsamp_model/opt_mae_snr'+str(snr),map_location=torch.device('cpu'))


# %% single image recon
seed = 0
with torch.no_grad():
    kspace, maps = test_data[0]  
    kspace = kspace.unsqueeze(0)
    maps = maps.unsqueeze(0)
    gt = toIm(kspace, maps).squeeze()
   
    
    # uni recon
    torch.manual_seed(seed=seed)
    kspace_noise = sample_uni(kspace)
    image_noise_uni = toIm(kspace_noise, maps).squeeze()
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_uni(image_input)
    image_recon = torch.cat((image_output[:,torch.arange(Nc),:,:].unsqueeze(4),image_output[:,torch.arange(Nc,2*Nc),:,:].unsqueeze(4)),4)
    image_uni = fastmri.complex_abs(torch.sum(fastmri.complex_mul(image_recon,fastmri.complex_conj(maps.to(device))),dim=1)).squeeze()

    # lwo50 recon
    torch.manual_seed(seed=seed)
    kspace_noise = sample_low50(kspace)
    image_noise_low50 = toIm(kspace_noise, maps).squeeze()
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_low50(image_input)
    image_recon = torch.cat((image_output[:,torch.arange(Nc),:,:].unsqueeze(4),image_output[:,torch.arange(Nc,2*Nc),:,:].unsqueeze(4)),4)
    image_low50 = fastmri.complex_abs(torch.sum(fastmri.complex_mul(image_recon,fastmri.complex_conj(maps.to(device))),dim=1)).squeeze()

    # lwo25 recon
    torch.manual_seed(seed=seed)
    kspace_noise = sample_low25(kspace)
    image_noise_low25 = toIm(kspace_noise, maps).squeeze()
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_low25(image_input)
    image_recon = torch.cat((image_output[:,torch.arange(Nc),:,:].unsqueeze(4),image_output[:,torch.arange(Nc,2*Nc),:,:].unsqueeze(4)),4)
    image_low25 = fastmri.complex_abs(torch.sum(fastmri.complex_mul(image_recon,fastmri.complex_conj(maps.to(device))),dim=1)).squeeze()

    # opt recon
    torch.manual_seed(seed=seed)
    kspace_noise = sample_opt(kspace)
    image_noise_opt = toIm(kspace_noise, maps).squeeze()
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_opt(image_input)
    image_recon = torch.cat((image_output[:,torch.arange(Nc),:,:].unsqueeze(4),image_output[:,torch.arange(Nc,2*Nc),:,:].unsqueeze(4)),4)
    image_opt = fastmri.complex_abs(torch.sum(fastmri.complex_mul(image_recon,fastmri.complex_conj(maps.to(device))),dim=1)).squeeze()

# %% save single image
#save_image(image_low25/gt.max()*2,'/home/wjy/Project/optsamp_result/low25_mse_snr10.png')

# %% save error map
#error = (image_low25-gt).abs()/gt.max()*5
#error = error.numpy()
#plt.imshow(error, cmap='hot',vmax=0.5,vmin=0.08)
#plt.colorbar()  # Optional colorbar
#plt.axis('off')
# Save the image
#plt.savefig('/home/wjy/Project/optsamp_result/low25_error_mse_snr10.png', bbox_inches='tight', pad_inches=0) 

# %% save patch
resize_transform = torchvision.transforms.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

#patch1 = resize_transform(gt[120:200,170:250].unsqueeze(0))/torch.max(gt)*2
#save_image(patch1,'/home/wjy/Project/optsamp_result/gt_p1.png')

#patch2 = resize_transform(gt[160:240,60:140].unsqueeze(0))/torch.max(gt)*2
#save_image(patch2,'/home/wjy/Project/optsamp_result/gt_p2.png')


# %%
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
ssim_module = SSIM(data_range=255, size_average=True, channel=1)

ssim_uni, ssim_opt, ssim_low50, ssim_low25 = 0, 0, 0, 0
nrmse_uni, nrmse_opt, nrmse_low50, nrmse_low25 = 0, 0, 0, 0 
nmae_uni, nmae_opt, nmae_low50, nmae_low25 = 0, 0, 0, 0

# %% recon
count = 0
with torch.no_grad():
  for kspace, maps in test_dataloader:
    count += 1
    gt = toIm(kspace, maps)
    scale = gt.max()
    l2scale = gt.norm(p=2)
    l1scale = gt.norm(p=1)

    # uni recon
    kspace_noise = sample_uni(kspace)
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_uni(image_input)
    image_recon = torch.cat((image_output[:,torch.arange(Nc),:,:].unsqueeze(4),image_output[:,torch.arange(Nc,2*Nc),:,:].unsqueeze(4)),4)
    image_uni = fastmri.complex_abs(torch.sum(fastmri.complex_mul(image_recon,fastmri.complex_conj(maps.to(device))),dim=1)).squeeze()
    ssim_uni += ssim_module(gt.unsqueeze(0).unsqueeze(1)/scale*256, image_uni.unsqueeze(0).unsqueeze(1)/scale*256)
    nrmse_uni += (image_uni-gt).norm(p=2)/l2scale
    nmae_uni += (image_uni-gt).norm(p=1)/l1scale

    # lwo50 recon
    kspace_noise = sample_low50(kspace)
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_low50(image_input)
    image_recon = torch.cat((image_output[:,torch.arange(Nc),:,:].unsqueeze(4),image_output[:,torch.arange(Nc,2*Nc),:,:].unsqueeze(4)),4)
    image_low50 = fastmri.complex_abs(torch.sum(fastmri.complex_mul(image_recon,fastmri.complex_conj(maps.to(device))),dim=1)).squeeze()
    ssim_low50 +=ssim_module(gt.unsqueeze(0).unsqueeze(1)/scale*256, image_low50.unsqueeze(0).unsqueeze(1)/scale*256)
    nrmse_low50 += (image_low50-gt).norm(p=2)/l2scale
    nmae_low50 += (image_low50-gt).norm(p=1)/l1scale

    # lwo25 recon
    kspace_noise = sample_low25(kspace)
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_low25(image_input)
    image_recon = torch.cat((image_output[:,torch.arange(Nc),:,:].unsqueeze(4),image_output[:,torch.arange(Nc,2*Nc),:,:].unsqueeze(4)),4)
    image_low25 = fastmri.complex_abs(torch.sum(fastmri.complex_mul(image_recon,fastmri.complex_conj(maps.to(device))),dim=1)).squeeze()
    ssim_low25 += ssim_module(gt.unsqueeze(0).unsqueeze(1)/scale*256, image_low25.unsqueeze(0).unsqueeze(1)/scale*256)
    nrmse_low25 += (image_low25-gt).norm(p=2)/l2scale
    nmae_low25 += (image_low25-gt).norm(p=1)/l1scale

    # opt recon
    kspace_noise = sample_opt(kspace)
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_opt(image_input)
    image_recon = torch.cat((image_output[:,torch.arange(Nc),:,:].unsqueeze(4),image_output[:,torch.arange(Nc,2*Nc),:,:].unsqueeze(4)),4)
    image_opt = fastmri.complex_abs(torch.sum(fastmri.complex_mul(image_recon,fastmri.complex_conj(maps.to(device))),dim=1)).squeeze()
    ssim_opt += ssim_module(gt.unsqueeze(0).unsqueeze(1)/scale*256, image_opt.unsqueeze(0).unsqueeze(1)/scale*256)
    nrmse_opt += (image_opt-gt).norm(p=2)/l2scale
    nmae_opt += (image_opt-gt).norm(p=1)/l1scale

print('ssim: ', 'uni',ssim_uni/count, ' low50',ssim_low50/count, ' low25',ssim_low25/count, ' opt',ssim_opt/count)
print('nrmse: ', 'uni',nrmse_uni/count, ' low50',nrmse_low50/count, ' low25',nrmse_low25/count, ' opt',nrmse_opt/count)
print('nmae: ', 'uni',nmae_uni/count, ' low50',nmae_low50/count, ' low25',nmae_low25/count, ' opt',nmae_opt/count)

# %%



