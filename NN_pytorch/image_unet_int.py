
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

#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
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

train_data = SliceDataset(
    #root=pathlib.Path('/home/wjy/Project/fastmri_dataset/brain_T1_demo/'),
    root = pathlib.Path('/project/jhaldar_118/jiayangw/dataset/brain_T1/multicoil_train/'),
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

class Sample(torch.nn.Module): 

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

def toIm(kspace,maps): 
    image = fastmri.complex_abs(torch.sum(fastmri.complex_mul(fastmri.ifft2c(kspace),fastmri.complex_conj(maps)),dim=1))
    return image.squeeze()

# %% sampling
factor = 8
snr = 5
sigma =  math.sqrt(8)*45/snr
print("SNR:", snr)
print('opt')

sample_model = Sample(sigma,factor)

# %% unet loader
recon_model = Unet(
  in_chans = 40,
  out_chans = 40,
  chans = 128,
  num_pool_layers = 4,
  drop_prob = 0.0
)

recon_model = torch.load("/project/jhaldar_118/jiayangw/OptSamp/model/opt_mae_snr"+str(snr))
weight = torch.load("/project/jhaldar_118/jiayangw/OptSamp/model/opt_mae_mask_snr"+str(snr))
sample_model.weight = weight

# %% data loader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dataloader = torch.utils.data.DataLoader(train_data,batch_size,shuffle=True)

sample_model.to(device)
recon_model.to(device)

# %% optimization parameters
recon_optimizer = optim.Adam(recon_model.parameters(),lr=3e-4)
L1Loss = torch.nn.L1Loss()
L2Loss = torch.nn.MSELoss()

# %% training
max_epochs = 10

for epoch in range(max_epochs):
    print("epoch:",epoch+1)

    batch_count = 0
    for kspace, maps in train_dataloader:

        batch_count = batch_count + 1

        gt = toIm(kspace, maps) # ground truth
        kspace_noise = sample_model(kspace) # add noise
        
        # forward
        image_noise = fastmri.ifft2c(kspace_noise)
        image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1).to(device)
        image_output = recon_model(image_input).to(device)
        image_recon = torch.cat((image_output[:,torch.arange(Nc),:,:].unsqueeze(4),image_output[:,torch.arange(Nc,2*Nc),:,:].unsqueeze(4)),4).to(device)
        
        recon = fastmri.complex_abs(torch.sum(fastmri.complex_mul(image_recon,fastmri.complex_conj(maps.to(device))),dim=1)).squeeze()


        loss = L1Loss(recon.to(device),gt.to(device))

        if batch_count%10 == 0:
            print("batch:",batch_count,"L1 loss:",loss.item())
        
        # backward
        loss.backward()
        
        # optimize network
        recon_optimizer.step()
        recon_optimizer.zero_grad()
    
    torch.save(recon_model,"/project/jhaldar_118/jiayangw/OptSamp/model/int_mae_snr"+str(snr))


# %%