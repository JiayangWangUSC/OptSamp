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
    root=pathlib.Path('/home/wjy/Project/fastmri_dataset/brain_T1/'),
    #root = pathlib.Path('/project/jhaldar_118/jiayangw/dataset/brain_T1/multicoil_train/'),
    transform=data_transform,
    challenge='multicoil'
)

# %% noise generator and transform to image
batch_size = 1

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

class Sample_opt75(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()

        self.weight = 1e-7*torch.ones(N2)
        self.weight[torch.arange(40,280)] = factor*4/3
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
    
class Sample_opt50(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()

        self.weight = 1e-7*torch.ones(N2)
        self.weight[torch.arange(80,240)] = 2*factor
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

class Sample_opt25(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()

        self.weight = 1e-7*torch.ones(N2)
        self.weight[torch.arange(120,200)] = 4*factor
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
snr = 5
sigma =  0.15*math.sqrt(8)/snr

# %%
sample_uni100 = Sample_uni100(sigma,factor)
sample_uni75 = Sample_uni75(sigma,factor)
sample_uni50 = Sample_uni50(sigma,factor)

recon_uni100 = torch.load('/home/wjy/Project/optsamp_model/uni100_mse_snr'+str(snr),map_location=torch.device('cpu'))
recon_uni75 = torch.load('/home/wjy/Project/optsamp_model/uni75_mse_snr'+str(snr),map_location=torch.device('cpu'))
recon_uni50 = torch.load('/home/wjy/Project/optsamp_model/uni50_mse_snr'+str(snr),map_location=torch.device('cpu'))

# %%
weight100 = torch.load('/home/wjy/Project/optsamp_model/opt100_mse_mask_snr'+str(snr))
sample_opt100 = Sample_opt100(sigma,factor)
sample_opt100.weight = weight100
weight75 = torch.load('/home/wjy/Project/optsamp_model/opt75_mse_mask_snr'+str(snr))
sample_opt75 = Sample_opt75(sigma,factor)
sample_opt75.weight = weight75
weight50 = torch.load('/home/wjy/Project/optsamp_model/opt50_mse_mask_snr'+str(snr))
sample_opt50 = Sample_opt50(sigma,factor)
sample_opt50.weight = weight50

recon_opt100 = torch.load('/home/wjy/Project/optsamp_model/opt100_mse_snr'+str(snr),map_location=torch.device('cpu'))
recon_opt75 = torch.load('/home/wjy/Project/optsamp_model/opt75_mse_snr'+str(snr),map_location=torch.device('cpu'))
recon_opt50 = torch.load('/home/wjy/Project/optsamp_model/opt50_mse_snr'+str(snr),map_location=torch.device('cpu'))

# %%
weight_mse = torch.load('/home/wjy/Project/optsamp_model/opt100_mse_mask_snr'+str(snr))
sample_opt_mse = Sample_opt100(sigma,factor)
sample_opt_mse.weight = weight_mse

weight_mae = torch.load('/home/wjy/Project/optsamp_model/opt100_mae_mask_snr'+str(snr))
sample_opt_mae = Sample_opt100(sigma,factor)
sample_opt_mae.weight = weight_mae

recon_optmse = torch.load('/home/wjy/Project/optsamp_model/opt100_mse_snr'+str(snr),map_location=torch.device('cpu'))
recon_optmae = torch.load('/home/wjy/Project/optsamp_model/opt100_mae_snr'+str(snr),map_location=torch.device('cpu'))

# %% single image recon
seed = 0
with torch.no_grad():
    kspace, maps = test_data[10]  
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
    

    # uni75 recon
    torch.manual_seed(seed=seed)
    kspace_noise = sample_uni75(kspace)
    image_noise_uni75 = toIm(kspace_noise, maps).squeeze()
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_uni75(image_input)
    recon = fastmri.complex_abs(torch.cat((image_output[:,0,:,:].unsqueeze(1).unsqueeze(4),image_output[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
    image_uni75 = recon * support.to(device)

    # uni50 recon
    torch.manual_seed(seed=seed)
    kspace_noise = sample_uni50(kspace)
    image_noise_uni50 = toIm(kspace_noise, maps).squeeze()
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_uni50(image_input)
    recon = fastmri.complex_abs(torch.cat((image_output[:,0,:,:].unsqueeze(1).unsqueeze(4),image_output[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
    image_uni50 = recon * support.to(device)

    # opt mse recon
    torch.manual_seed(seed=seed)
    kspace_noise = sample_opt_mse(kspace)
    image_noise_optmse = toIm(kspace_noise, maps).squeeze()
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_optmse(image_input)
    recon = fastmri.complex_abs(torch.cat((image_output[:,0,:,:].unsqueeze(1).unsqueeze(4),image_output[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
    image_optmse = recon * support.to(device)

    # opt mae recon
    torch.manual_seed(seed=seed)
    kspace_noise = sample_opt_mae(kspace)
    image_noise_optmae = toIm(kspace_noise, maps).squeeze()
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1)
    image_output = recon_optmae(image_input)
    recon = fastmri.complex_abs(torch.cat((image_output[:,0,:,:].unsqueeze(1).unsqueeze(4),image_output[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
    image_optmae = recon * support.to(device)

# %% save single image
save_image(gt/gt.max()*2,'/home/wjy/Project/optsamp_result/gt.png')
save_image(image_uni100/gt.max()*2,'/home/wjy/Project/optsamp_result/uni100_snr5.png')
save_image(image_uni75/gt.max()*2,'/home/wjy/Project/optsamp_result/uni75_snr5.png')
save_image(image_uni50/gt.max()*2,'/home/wjy/Project/optsamp_result/uni50_snr5.png')

save_image(image_optmse/gt.max()*2,'/home/wjy/Project/optsamp_result/optmse_snr5.png')
save_image(image_optmae/gt.max()*2,'/home/wjy/Project/optsamp_result/optmae_snr5.png')

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

ssim_opt100, ssim_opt75, ssim_opt50, ssim_opt25 = 0, 0, 0, 0
nrmse_opt100, nrmse_opt75, nrmse_opt50, nrmse_opt25 = 0, 0, 0, 0 

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

    # uni75 recon
    kspace_uni75 = sample_uni75(kspace)
    noise_uni75 = fastmri.ifft2c(kspace_uni75)
    input_uni75 = torch.cat((noise_uni75[:,:,:,:,0],noise_uni75[:,:,:,:,1]),1).to(device)
    output_uni75 = recon_uni75(input_uni75).to(device)
    image_uni75 = support*fastmri.complex_abs(torch.cat((output_uni75[:,0,:,:].unsqueeze(1).unsqueeze(4),output_uni75[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
    ssim_uni75 += ssim_module(image_uni75.unsqueeze(0).unsqueeze(1)/scale*256, gt.unsqueeze(0).unsqueeze(1)/scale*256)
    nrmse_uni75 += (image_uni75-gt).norm(p=2)/l2scale

    # uni50 recon
    kspace_uni50 = sample_uni50(kspace)
    noise_uni50 = fastmri.ifft2c(kspace_uni50)
    input_uni50 = torch.cat((noise_uni50[:,:,:,:,0],noise_uni50[:,:,:,:,1]),1).to(device)
    output_uni50 = recon_uni50(input_uni50).to(device)
    image_uni50 = support*fastmri.complex_abs(torch.cat((output_uni50[:,0,:,:].unsqueeze(1).unsqueeze(4),output_uni50[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
    ssim_uni50 += ssim_module(image_uni50.unsqueeze(0).unsqueeze(1)/scale*256, gt.unsqueeze(0).unsqueeze(1)/scale*256)
    nrmse_uni50 += (image_uni50-gt).norm(p=2)/l2scale

print('ssim: ', 'uni100',ssim_uni100/count, ' uni75',ssim_uni75/count, ' uni50',ssim_uni50/count)
print('nrmse: ', 'uni100',nrmse_uni100/count, ' uni75',nrmse_uni75/count, ' uni50',nrmse_uni50/count)

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

    # opt100 recon
    kspace_opt100 = sample_opt100(kspace)
    noise_opt100 = fastmri.ifft2c(kspace_opt100)
    input_opt100 = torch.cat((noise_opt100[:,:,:,:,0],noise_opt100[:,:,:,:,1]),1).to(device)
    output_opt100 = recon_opt100(input_opt100).to(device)
    image_opt100 = support*fastmri.complex_abs(torch.cat((output_opt100[:,0,:,:].unsqueeze(1).unsqueeze(4),output_opt100[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
    ssim_opt100 += ssim_module(image_opt100.unsqueeze(0).unsqueeze(1)/scale*256, gt.unsqueeze(0).unsqueeze(1)/scale*256)
    nrmse_opt100 += (image_opt100-gt).norm(p=2)/l2scale

    # opt75 recon
    kspace_opt75 = sample_opt75(kspace)
    noise_opt75 = fastmri.ifft2c(kspace_opt75)
    input_opt75 = torch.cat((noise_opt75[:,:,:,:,0],noise_opt75[:,:,:,:,1]),1).to(device)
    output_opt75 = recon_opt75(input_opt75).to(device)
    image_opt75 = support*fastmri.complex_abs(torch.cat((output_opt75[:,0,:,:].unsqueeze(1).unsqueeze(4),output_opt75[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
    ssim_opt75 += ssim_module(image_opt75.unsqueeze(0).unsqueeze(1)/scale*256, gt.unsqueeze(0).unsqueeze(1)/scale*256)
    nrmse_opt75 += (image_opt75-gt).norm(p=2)/l2scale

    # opt50 recon
    kspace_opt50 = sample_opt50(kspace)
    noise_opt50 = fastmri.ifft2c(kspace_opt50)
    input_opt50 = torch.cat((noise_opt50[:,:,:,:,0],noise_opt50[:,:,:,:,1]),1).to(device)
    output_opt50 = recon_opt50(input_opt50).to(device)
    image_opt50 = support*fastmri.complex_abs(torch.cat((output_opt50[:,0,:,:].unsqueeze(1).unsqueeze(4),output_opt50[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
    ssim_opt50 += ssim_module(image_opt50.unsqueeze(0).unsqueeze(1)/scale*256, gt.unsqueeze(0).unsqueeze(1)/scale*256)
    nrmse_opt50 += (image_opt50-gt).norm(p=2)/l2scale

print('ssim: ', 'opt100',ssim_opt100/count, ' opt75',ssim_opt75/count, ' opt50',ssim_opt50/count)
print('nrmse: ', 'opt100',nrmse_opt100/count, ' opt75',nrmse_opt75/count, ' opt50',nrmse_opt50/count)


# %%
