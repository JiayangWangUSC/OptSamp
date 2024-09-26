
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

def toIm(kspace,maps): 
    # kspace-(batch,Nc,N1,N2,2) maps-(batch,Nc,N1,N2,2)
    # image-(batch,N1,N2)
    image = fastmri.complex_abs(torch.sum(fastmri.complex_mul(fastmri.ifft2c(kspace),fastmri.complex_conj(maps)),dim=1))
    return image.squeeze()

# %% sampling
factor = 8
snr = 5
sigma =  math.sqrt(8)*45/snr
print("SNR:", snr)
print('opt')

sample_model = Sample(sigma,factor)
weight = torch.load("/project/jhaldar_118/jiayangw/OptSamp/model/opt_mae_mask_snr"+str(snr))
sample_model.weight = weight

# %% unet loader
recon_model = Unet(
  in_chans = 40,
  out_chans = 40,
  chans = 32,
  num_pool_layers = 3,
  drop_prob = 0.0
)

#recon_model = torch.load("/project/jhaldar_118/jiayangw/OptSamp/model/opt_mae_snr"+str(snr))

# %% data loader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataloader = torch.utils.data.DataLoader(train_data,batch_size,shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_data,batch_size,shuffle=True)

sample_model.to(device)
recon_model.to(device)

# %% optimization parameters
recon_optimizer = optim.Adam(recon_model.parameters(),lr=3e-4)
L1Loss = torch.nn.L1Loss()
L2Loss = torch.nn.MSELoss()

#step = 0.3 # sampling weight optimization step size
print('L1 Loss')

# %% training
max_epochs = 200
#val_loss = torch.zeros(max_epochs)
for epoch in range(max_epochs):
    print("epoch:",epoch+1)
    #step = step * 0.99
    batch_count = 0
    trainloss = 0
    for kspace, maps in train_dataloader:
        
        batch_count = batch_count + 1
        #sample_model.weight.requires_grad = True

        gt = toIm(kspace, maps) # ground truth
        kspace_noise = sample_model(kspace) # add noise
        
        # forward
        image_noise = fastmri.ifft2c(kspace_noise)
        image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1).to(device)
        image_output = recon_model(image_input).to(device)
        image_recon = torch.cat((image_output[:,torch.arange(Nc),:,:].unsqueeze(4),image_output[:,torch.arange(Nc,2*Nc),:,:].unsqueeze(4)),4).to(device)
        
        recon = fastmri.complex_abs(torch.sum(fastmri.complex_mul(image_recon,fastmri.complex_conj(maps.to(device))),dim=1)).squeeze()


        loss = L1Loss(recon.to(device),gt.to(device))
        trainloss += loss.item()

        #if batch_count%10 == 0:
        #    print("batch:",batch_count,"L1 loss:",loss.item())
        
        # backward
        loss.backward()
        
        # optimize mask
        #with torch.no_grad():

        #    weight = sample_model.weight.clone() 
        #    ind = torch.where(weight >= 1)[0]

        #    grad = sample_model.weight.grad
        #    temp = grad[ind]
        #    temp = temp - temp.mean()
        #    temp = temp/temp.norm()
        #    grad = 0 * grad
        #    grad[ind] = temp

        #    weight = weight - step * grad
        #    weight[weight<1] = 1e-7
        #    ind = torch.where(weight >= 1)[0]
            
        #    temp = weight[ind] - 1
        #    temp = temp/temp.mean()*(factor*N2/len(ind)-1)
        #    weight[ind] = temp + 1

            #print("max:",weight.max(),"min:",weight.min(),"mean:", weight.mean())

        #    sample_model.weight = weight
        
        # optimize network
        recon_optimizer.step()
        recon_optimizer.zero_grad()

#    print("weight max:",weight.max(),"min:",weight.min(),"mean:", weight.mean())
    
    torch.save(recon_model,"/project/jhaldar_118/jiayangw/OptSamp/model/opt_mae_snr"+str(snr))
#    torch.save(weight,"/project/jhaldar_118/jiayangw/OptSamp/model/opt_mae_mask_snr"+str(snr))

    with torch.no_grad():
        valloss = 0
        for kspace, maps in val_dataloader:
            recon_model.eval()
            gt = toIm(kspace, maps)
        
            kspace_noise = sample_model(kspace)
            image_noise = fastmri.ifft2c(kspace_noise)
            image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1).to(device)
            image_output = recon_model(image_input).to(device)
            image_recon = torch.cat((image_output[:,torch.arange(Nc),:,:].unsqueeze(4),image_output[:,torch.arange(Nc,2*Nc),:,:].unsqueeze(4)),4).to(device)
        
            recon = fastmri.complex_abs(torch.sum(fastmri.complex_mul(image_recon,fastmri.complex_conj(maps.to(device))),dim=1)).squeeze()
            valloss += L1Loss(recon.to(device),gt.to(device))

    print("train loss:",trainloss/320," val loss:",valloss/48)

# %%
