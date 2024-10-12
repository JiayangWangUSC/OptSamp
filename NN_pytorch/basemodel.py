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

train_data = SliceDataset(
    #root=pathlib.Path('/home/wjy/Project/fastmri_dataset/brain_T1_demo/'),
    root = pathlib.Path('/project/jhaldar_118/jiayangw/dataset/brain_T1/multicoil_train/'),
    transform=data_transform,
    challenge='multicoil'
)

val_data = SliceDataset(
    #root=pathlib.Path('/home/wjy/Project/fastmri_dataset/brain_T1_demo/'),
    root = pathlib.Path('/project/jhaldar_118/jiayangw/dataset/brain_T1/multicoil_val/'),
    transform=data_transform,
    challenge='multicoil'
)


# %% noise generator and transform to image
batch_size = 8

# %% sampling
factor = 8

# %% unet loader
recon_model = Unet(
  in_chans = 32,
  out_chans = 32,
  chans = 64,
  num_pool_layers = 3,
  drop_prob = 0.0
)

recon_model = torch.load('/project/jhaldar_118/jiayangw/OptSamp/model/basemodel')
# %% GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataloader = torch.utils.data.DataLoader(train_data,batch_size,shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_data,batch_size,shuffle=True)

recon_model.to(device)

# %% optimizer
recon_optimizer = optim.Adam(recon_model.parameters(),lr=1e-2)

L2Loss = torch.nn.MSELoss()

snr = 50
sigma =  0.15/snr

# %% training
max_epochs = 10

for epoch in range(max_epochs):
    print("epoch:",epoch+1)

    trainloss = 0
    for kspace, maps in train_dataloader:

        noise = fastmri.ifft2c(sigma*torch.randn_like(kspace))
        noise = torch.cat((noise[:,:,:,:,0],noise[:,:,:,:,1]),1).to(device)
        image_input = fastmri.ifft2c(kspace)
        image_input = torch.cat((image_input[:,:,:,:,0],image_input[:,:,:,:,1]),1).to(device)
        image_output = recon_model(image_input+noise).to(device)

        loss = L2Loss(image_output.to(device),image_input.to(device))
        trainloss += loss.item()

        loss.backward()

        recon_optimizer.step()
        recon_optimizer.zero_grad()

    torch.save(recon_model,"/project/jhaldar_118/jiayangw/OptSamp/model/basemodel")

    with torch.no_grad():
        valloss = 0
        for kspace, maps in val_dataloader:
            recon_model.eval()
        
            noise = fastmri.ifft2c(sigma*torch.randn_like(kspace))
            noise = torch.cat((noise[:,:,:,:,0],noise[:,:,:,:,1]),1).to(device)
            image_input = fastmri.ifft2c(kspace)
            image_input = torch.cat((image_input[:,:,:,:,0],image_input[:,:,:,:,1]),1).to(device)
            image_output = recon_model(image_input+noise).to(device)

            
            valloss += L2Loss(image_output.to(device),image_input.to(device))

    print("train loss:",trainloss/331/8," val loss:",valloss/42/8)
