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
snr = 2
reso = 0
print('non-uniform')
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
print('opt', flush = True)

class Sample(torch.nn.Module): 

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

# %% sampling
factor = 8
sigma =  0.12*math.sqrt(8)/snr

sample_model = Sample(sigma,factor)
weight = torch.load("/project/jhaldar_118/jiayangw/OptSamp/model/opt_mse_mask_snr"+str(snr)+"_reso"+str(reso))
sample_model.weight = weight

# %% unet loader
recon_model = Unet(
  in_chans = 32,
  out_chans = 2,
  chans = 64,
  num_pool_layers = 3,
  drop_prob = 0.0
)

recon_model = torch.load("/project/jhaldar_118/jiayangw/OptSamp/model/opt_mse_snr"+str(snr)+"_reso"+str(reso))

# %% data loader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataloader = torch.utils.data.DataLoader(train_data,batch_size,shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_data,batch_size,shuffle=True)


sample_model.to(device)
recon_model.to(device)

# %% optimization parameters
recon_optimizer = optim.Adam(recon_model.parameters(),lr=1e-4)
print('L2 Loss', flush = True)
#Loss = torch.nn.L1Loss()
Loss = torch.nn.MSELoss()

step = 1e-2

# %% training
max_epochs = 10
for epoch in range(max_epochs):
    print("epoch:",epoch+1)
    if epoch < 0:
        step = 0.9 * step
        trainloss = 0
        trainloss_normalized = 0
        for kspace, maps in train_dataloader:
            sample_model.weight.requires_grad = True

            gt = toIm(kspace, maps) # ground truth
            support = fastmri.complex_abs(torch.sum(fastmri.complex_mul(maps,fastmri.complex_conj(maps)),dim=1))
            kspace_noise = sample_model(kspace) # add noise
        
            # forward
            image_noise = fastmri.ifft2c(kspace_noise)
            image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1).to(device)
            image_output = recon_model(image_input).to(device)
            recon = fastmri.complex_abs(torch.cat((image_output[:,0,:,:].unsqueeze(1).unsqueeze(4),image_output[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
            recon = recon * support.to(device)

            loss = Loss(recon.to(device),gt.to(device))
            trainloss += loss.item()
            trainloss_normalized += loss.item()/Loss(0*gt,gt)

            # backward
            loss.backward()
        
            # optimize mask
            with torch.no_grad():
                weight = sample_model.weight.clone() 
                grad = sample_model.weight.grad
                grad = grad - grad.mean()
                grad = grad/grad.norm()
                weight = weight - step * grad

                sample_model.weight = weight
        
            # optimize network
            recon_optimizer.step()
            recon_optimizer.zero_grad()

            print("weight max:",weight.max(),"weight min:",weight.min(), flush = True)

    else:
        sample_model.weight.requires_grad = False
        trainloss = 0
        trainloss_normalized = 0
        for kspace, maps in train_dataloader:

            gt = toIm(kspace, maps) # ground truth
            support = fastmri.complex_abs(torch.sum(fastmri.complex_mul(maps,fastmri.complex_conj(maps)),dim=1))
            kspace_noise = sample_model(kspace) # add noise
        
            # forward
            image_noise = fastmri.ifft2c(kspace_noise)
            image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1).to(device)
            image_output = recon_model(image_input).to(device)
            recon = fastmri.complex_abs(torch.cat((image_output[:,0,:,:].unsqueeze(1).unsqueeze(4),image_output[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
            recon = recon * support.to(device)

            loss = Loss(recon.to(device),gt.to(device))
            trainloss += loss.item()
            trainloss_normalized += loss.item()/Loss(0*gt,gt)

            # backward
            loss.backward()
            recon_optimizer.step()
            recon_optimizer.zero_grad()

    with torch.no_grad():
        valloss = 0
        valloss_normalized = 0
        for kspace, maps in val_dataloader:
            recon_model.eval()
            gt = toIm(kspace, maps)
            support = fastmri.complex_abs(torch.sum(fastmri.complex_mul(maps,fastmri.complex_conj(maps)),dim=1))
        
            kspace_noise = sample_model(kspace)
            image_noise = fastmri.ifft2c(kspace_noise)
            image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1).to(device)
            image_output = recon_model(image_input).to(device)
            recon = fastmri.complex_abs(torch.cat((image_output[:,0,:,:].unsqueeze(1).unsqueeze(4),image_output[:,1,:,:].unsqueeze(1).unsqueeze(4)),4)).squeeze().to(device)
            recon = recon * support.to(device)

            loss = Loss(recon.to(device),gt.to(device))
            valloss += loss.item()
            valloss_normalized += loss.item()/Loss(0*gt,gt)

    print("train loss:",trainloss/331/8," val loss:",valloss/42/8, flush = True)
    print("normalized train loss:",trainloss_normalized/331/8," normalized val loss:",valloss_normalized/42/8, flush = True)
    
    torch.save(recon_model,"/project/jhaldar_118/jiayangw/OptSamp/model/opt_mse_snr"+str(snr)+"_reso"+str(reso))
    torch.save(weight,"/project/jhaldar_118/jiayangw/OptSamp/model/opt_mse_mask_snr"+str(snr)+"_reso"+str(reso))

# %%
