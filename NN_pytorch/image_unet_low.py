
# %%
import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms
from fastmri.models import Unet

import numpy as np
import math
import pathlib
import torch.optim as optim
from fastmri.data import  mri_data

#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# %% data loader
def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the kspace to tensor format
    kspace = transforms.to_tensor(kspace)
    image = fastmri.ifft2c(kspace)
    image = image[:,torch.arange(191,575),:,:]
    kspace = fastmri.fft2c(image)/5e-5
    return kspace

train_data = mri_data.SliceDataset(
    #root=pathlib.Path('/home/wjy/Project/fastmri_dataset/test/'),
    root = pathlib.Path('/project/jhaldar_118/jiayangw/dataset/brain/train/'),
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
        self.factor = factor
        self.sigma = sigma/np.sqrt(2*16)

    def forward(self,kspace):
        noise = self.sigma*torch.randn_like(kspace)
        support = torch.zeros(396)
        # low_80(40,317)
        # low_60(80,237)
        # low_40(120,157)
        noise = noise/math.sqrt(self.factor/157*396)
        support[torch.arange(120,120+157)] = 1
        kspace_noise = torch.mul(kspace + noise, support.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(0).repeat(kspace.size(0),16,384,1,2))
        return kspace_noise

def toIm(kspace): 
    image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1)
    return image

# %% sampling
factor = 8
sigma = 8
print("noise level:", sigma)
sample_model = Sample(sigma,factor)

# %% unet loader
recon_model = Unet(
  in_chans = 32,
  out_chans = 32,
  chans = 128,
  num_pool_layers = 4,
  drop_prob = 0.0
)

#recon_model = torch.load('/home/wjy/Project/optsamp_models/uni_model_noise0.3',map_location=torch.device('cpu'))

# %% GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataloader = torch.utils.data.DataLoader(train_data,batch_size,shuffle=True)
#val_dataloader = torch.utils.data.DataLoader(val_data,batch_size,shuffle=True)

sample_model.to(device)
recon_model.to(device)

# %% optimizer
recon_optimizer = optim.Adam(recon_model.parameters(),lr=3e-4)
#Loss = torch.nn.MSELoss()
L1Loss = torch.nn.L1Loss()
#L2Loss = torch.nn.MSELoss()
#beta = 1e-3
#ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=1)

# %% training
max_epochs = 50
#val_loss = torch.zeros(max_epochs)
for epoch in range(max_epochs):
    print("epoch:",epoch+1)
    batch_count = 0
    for train_batch in train_dataloader:
        
        batch_count = batch_count + 1
        train_batch.to(device)
        gt = toIm(train_batch)
        #support = torch.ge(gt,0.05*torch.max(gt))
        
        kspace_noise = sample_model(train_batch).to(device)
        image_noise = fastmri.ifft2c(kspace_noise).to(device)
        image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1).to(device) 
        image_output = recon_model(image_input).to(device)
        image_recon = torch.cat((image_output[:,torch.arange(16),:,:].unsqueeze(4),image_output[:,torch.arange(16,32),:,:].unsqueeze(4)),4).to(device)
        recon = fastmri.rss(fastmri.complex_abs(image_recon), dim=1)
        #loss = L2Loss(torch.mul(recon.to(device),support.to(device)),torch.mul(gt.to(device),support.to(device))) + beta*L1Loss(torch.mul(recon.to(device),gradmap.to(device)),torch.mul(gt.to(device),gradmap.to(device)))
        #loss = L1Loss(torch.mul(recon.to(device),support.to(device)),torch.mul(gt.to(device),support.to(device)))
        loss = L1Loss(recon.to(device),gt.to(device))

        if batch_count%100 == 0:
            print("batch:",batch_count,"train loss:",loss.item())
        
        loss.backward()
        recon_optimizer.step()
        recon_optimizer.zero_grad()

    torch.save(recon_model,"/project/jhaldar_118/jiayangw/OptSamp/model/low40_L1_model_sigma"+str(sigma))

