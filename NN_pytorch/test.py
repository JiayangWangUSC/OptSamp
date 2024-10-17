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

class Sample_opt100(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.weight = factor*torch.ones(N2)
        self.factor = factor
        self.sigma = sigma

    def forward(self,kspace):
    
        support = self.weight >= 1
        support = support.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(4).repeat(kspace.size(0),Nc,N1,1,2)

        mask = self.weight.clone() 
        mask = 1.0 / (self.weight ** 0.5)
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(4).repeat(kspace.size(0),Nc,N1,1,2)
        
        
        noise = self.sigma*torch.randn_like(kspace)
        kspace_noise = support * (kspace + mask * noise) 
        return kspace_noise

class Sample_uni100(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.mask = factor*torch.ones(N2)
        self.sigma = sigma

    def forward(self,kspace):
        noise = self.sigma*torch.randn_like(kspace)
        kspace_noise = kspace + torch.div(noise,torch.sqrt(self.mask).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(4).repeat(kspace.size(0),Nc,N1,1,2))  # need to reshape mask        image = fastmri.ifft2c(kspace_noise)
        return kspace_noise
    
class Sample_uni75(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.mask = factor*torch.ones(N2)
        self.sigma = sigma

    def forward(self,kspace):
        noise = self.sigma*torch.randn_like(kspace)
        
        support = torch.zeros(N2)
        support[torch.arange(40,280)] = 1
        noise = noise/math.sqrt(factor*4/3)
        
        kspace_noise = torch.mul(kspace + noise, support.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(0).repeat(kspace.size(0),Nc,N1,1,2))
        
        return kspace_noise

class Sample_uni50(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.mask = factor*torch.ones(N2)
        self.sigma = sigma

    def forward(self,kspace):
        noise = self.sigma*torch.randn_like(kspace)
        
        support = torch.zeros(N2)
        support[torch.arange(80,240)] = 1
        noise = noise/math.sqrt(factor*2)
        
        kspace_noise = torch.mul(kspace + noise, support.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(0).repeat(kspace.size(0),Nc,N1,1,2))
        
        return kspace_noise

class Sample_uni25(torch.nn.Module): 

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


def toIm(kspace,maps): 
    # kspace-(batch,Nc,N1,N2,2) maps-(batch,Nc,N1,N2,2)
    # image-(batch,N1,N2)
    image = fastmri.complex_abs(torch.sum(fastmri.complex_mul(fastmri.ifft2c(kspace),fastmri.complex_conj(maps)),dim=1))
    return image.squeeze()

# %% GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_dataloader = torch.utils.data.DataLoader(test_data,batch_size,shuffle=True)

# %% parameters
factor = 8
snr = 10
sigma =  0.15*math.sqrt(8)/snr


# %%
#weight = torch.load('/home/wjy/Project/optsamp_model/opt_mse_mask_snr'+str(snr))

sample_uni100 = Sample_uni100(sigma,factor)
sample_uni75 = Sample_uni75(sigma,factor)
sample_uni50 = Sample_uni50(sigma,factor)
sample_uni25 = Sample_uni25(sigma,factor)

recon_uni100 = torch.load('/home/wjy/Project/optsamp_model/uni100_mse_snr'+str(snr),map_location=torch.device('cpu'))
recon_uni75 = torch.load('/home/wjy/Project/optsamp_model/uni75_mse_snr'+str(snr),map_location=torch.device('cpu'))
recon_uni50 = torch.load('/home/wjy/Project/optsamp_model/uni50_mse_snr'+str(snr),map_location=torch.device('cpu'))
recon_uni25 = torch.load('/home/wjy/Project/optsamp_model/uni25_mse_snr'+str(snr),map_location=torch.device('cpu'))
#recon_opt = torch.load('/home/wjy/Project/optsamp_model/opt_mse_snr'+str(snr),map_location=torch.device('cpu'))


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
#resize_transform = torchvision.transforms.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

#patch1 = resize_transform(gt[120:200,170:250].unsqueeze(0))/torch.max(gt)*2
#save_image(patch1,'/home/wjy/Project/optsamp_result/gt_p1.png')

#patch2 = resize_transform(gt[160:240,60:140].unsqueeze(0))/torch.max(gt)*2
#save_image(patch2,'/home/wjy/Project/optsamp_result/gt_p2.png')


# %%
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
ssim_module = SSIM(data_range=255, size_average=True, channel=1)

ssim_uni100, ssim_uni75, ssim_uni50, ssim_uni25 = 0, 0, 0, 0
nrmse_uni100, nrmse_uni75, nrmse_uni50, nrmse_uni25 = 0, 0, 0, 0 
nmae_uni100, nmae_uni75, nmae_uni50, nmae_uni25 = 0, 0, 0, 0

# %% recon
count = 0
with torch.no_grad():
  for kspace, maps in test_dataloader:
    count += 1
    gt = toIm(kspace, maps)
    support = fastmri.complex_abs(torch.sum(fastmri.complex_mul(maps,fastmri.complex_conj(maps)),dim=1)).squeeze()
    scale = gt.max()
    l2scale = gt.norm(p=2)
    l1scale = gt.norm(p=1)

    # uni100 recon
    kspace_uni100 = sample_uni100(kspace)
    noise_uni100 = fastmri.ifft2c(kspace_uni100)
    input_uni100 = torch.cat((noise_uni100[:,:,:,:,0],noise_uni100[:,:,:,:,1]),1).to(device)
    output_uni100 = recon_uni100(input_uni100).to(device)
    image_uni100 = support*fastmri.complex_abs(torch.cat((output_uni100[:,0,:,:].unsqueeze(1).unsqueeze(4),output_uni100[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
    ssim_uni100 += ssim_module(image_uni100.unsqueeze(0).unsqueeze(1)/scale*256, gt.unsqueeze(0).unsqueeze(1)/scale*256)
    nrmse_uni100 += (image_uni100-gt).norm(p=2)/l2scale
    nmae_uni100 += (image_uni100-gt).norm(p=1)/l1scale

    # uni75 recon
    kspace_uni75 = sample_uni75(kspace)
    noise_uni75 = fastmri.ifft2c(kspace_uni75)
    input_uni75 = torch.cat((noise_uni75[:,:,:,:,0],noise_uni75[:,:,:,:,1]),1).to(device)
    output_uni75 = recon_uni75(input_uni75).to(device)
    image_uni75 = support*fastmri.complex_abs(torch.cat((output_uni75[:,0,:,:].unsqueeze(1).unsqueeze(4),output_uni75[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
    ssim_uni75 += ssim_module(image_uni75.unsqueeze(0).unsqueeze(1)/scale*256, gt.unsqueeze(0).unsqueeze(1)/scale*256)
    nrmse_uni75 += (image_uni75-gt).norm(p=2)/l2scale
    nmae_uni75 += (image_uni75-gt).norm(p=1)/l1scale

    # uni50 recon
    kspace_uni50 = sample_uni50(kspace)
    noise_uni50 = fastmri.ifft2c(kspace_uni50)
    input_uni50 = torch.cat((noise_uni50[:,:,:,:,0],noise_uni50[:,:,:,:,1]),1).to(device)
    output_uni50 = recon_uni50(input_uni50).to(device)
    image_uni50 = support*fastmri.complex_abs(torch.cat((output_uni50[:,0,:,:].unsqueeze(1).unsqueeze(4),output_uni50[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
    ssim_uni50 += ssim_module(image_uni50.unsqueeze(0).unsqueeze(1)/scale*256, gt.unsqueeze(0).unsqueeze(1)/scale*256)
    nrmse_uni50 += (image_uni50-gt).norm(p=2)/l2scale
    nmae_uni50 += (image_uni50-gt).norm(p=1)/l1scale

    # uni25 recon
    kspace_uni25 = sample_uni25(kspace)
    noise_uni25 = fastmri.ifft2c(kspace_uni25)
    input_uni25 = torch.cat((noise_uni25[:,:,:,:,0],noise_uni25[:,:,:,:,1]),1).to(device)
    output_uni25 = recon_uni25(input_uni25).to(device)
    image_uni25 = support*fastmri.complex_abs(torch.cat((output_uni25[:,0,:,:].unsqueeze(1).unsqueeze(4),output_uni25[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
    ssim_uni25 += ssim_module(image_uni25.unsqueeze(0).unsqueeze(1)/scale*256, gt.unsqueeze(0).unsqueeze(1)/scale*256)
    nrmse_uni25 += (image_uni25-gt).norm(p=2)/l2scale
    nmae_uni25 += (image_uni25-gt).norm(p=1)/l1scale

    # opt recon
    #kspace_noise = sample_opt(kspace)
    #image_noise = fastmri.ifft2c(kspace_noise)
    #image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1).to(device)
    #image_output = recon_opt(image_input).to(device)
    #image_opt = support*fastmri.complex_abs(torch.cat((image_output[:,0,:,:].unsqueeze(1).unsqueeze(4),image_output[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
    #ssim_opt += ssim_module(gt.unsqueeze(0).unsqueeze(1)/scale*256, image_opt.unsqueeze(0).unsqueeze(1)/scale*256)
    #nrmse_opt += (image_opt-gt).norm(p=2)/l2scale
    #nmae_opt += (image_opt-gt).norm(p=1)/l1scale

print('ssim: ', 'uni100',ssim_uni100/count, ' uni75',ssim_uni75/count, ' uni50',ssim_uni50/count, ' uni25',ssim_uni25/count,)
print('nrmse: ', 'uni100',nrmse_uni100/count, ' uni75',nrmse_uni75/count, ' uni50',nrmse_uni50/count, ' uni25',nrmse_uni25/count,)
print('nmae: ', 'uni100',nmae_uni100/count,  ' uni75',nmae_uni75/count, ' uni50',nmae_uni50/count, ' uni25',nmae_uni25/count,)

# %%



