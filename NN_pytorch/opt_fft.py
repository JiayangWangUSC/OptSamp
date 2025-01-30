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
reso = 3
print('optimized fft')
print("SNR:", snr, flush = True)
print('resolution:', reso, flush = True)

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
        self.weight =  torch.ones(N1-32*reso) * torch.ones(N2-32*reso)

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

sample_model = Sample(sigma,factor)
recon_model = Recon()

# %% data loader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataloader = torch.utils.data.DataLoader(train_data,batch_size,shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_data,batch_size,shuffle=True)

sample_model.to(device)
recon_model.to(device)

# %% optimization parameters
Loss = torch.nn.MSELoss()
step1 = 10
step2 = 0.1

# %% training
max_epochs = 10
for epoch in range(max_epochs):
    print("epoch:",epoch+1)

    step1 = 0.9 * step1
    step2 = 0.9 * step2
    #trainloss = 0
    #trainloss_normalized = 0
    for kspace, maps in train_dataloader:
        sample_model.weight.requires_grad = True
        recon_model.weight.requires_grad = True
        gt = toIm(kspace, maps) # ground truth

        kspace_noise = recon_model(sample_model(kspace)) # add noise and apply window
        recon = toIm(kspace_noise, maps)

        loss = Loss(recon.to(device),gt.to(device))
        #trainloss += loss.item()
        #trainloss_normalized += loss.item()/Loss(0*gt,gt)

        # backward
        loss.backward()
        
        # optimize mask
        with torch.no_grad():
            weight1 = recon_model.weight.clone() 
            grad = recon_model.weight.grad
            weight1 = weight1 - step1 * grad
            weight1[weight1 > 1] = 1
            weight1[weight1 < 0] = 0
            recon_model.weight = weight1

            weight2 = sample_model.weight.clone() 
            grad = sample_model.weight.grad
            grad = grad - grad.mean()
            grad = grad/grad.norm()
            weight2 = weight2 - step2 * grad
            sample_model.weight = weight2
        
        print("window weight max:", weight1.max(), " min:", weight1.min(), flush = True)
        print("sampling weight max:", weight2.max(), " min:", weight2.min(), flush = True)

    torch.save(weight1, "/project/jhaldar_118/jiayangw/OptSamp/model/uni_window_snr"+str(snr)+"_reso"+str(reso))

# %%
