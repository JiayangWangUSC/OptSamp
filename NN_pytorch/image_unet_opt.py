
# %%
import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms
from fastmri.models import Unet
import pathlib
import numpy as np
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
    #root=pathlib.Path('/home/wjy/Project/fastmri_dataset/miniset_brain/'),
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

class Sample(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.mask = 2*torch.rand(396)-1
        self.factor = factor
        self.sigma = sigma/np.sqrt(2*16)

    def forward(self,kspace):
        sample_mask = torch.sqrt(1 + F.softmax(self.mask)*(self.factor-1)*396)
        noise = self.sigma*torch.randn_like(kspace)
        kspace_noise = kspace + torch.div(noise,sample_mask.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(0).repeat(kspace.size(0),16,384,1,2)) 
        return kspace_noise

def toIm(kspace): 
    image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1)
    return image

# %% sampling and noise level parameters
factor = 8
sigma = 8 #1,2,4,8
print("noise level:", sigma)

# %% unet loader
recon_model = Unet(
  in_chans = 32,
  out_chans = 32,
  chans = 128,
  num_pool_layers = 4,
  drop_prob = 0.0
)

recon_model = torch.load('/project/jhaldar_118/jiayangw/OptSamp/model/uni_model_sigma'+str(sigma))
#recon_model = torch.load('/home/wjy/Project/optsamp_models/uni_model_noise0.3',map_location=torch.device('cpu'))

# %% data loader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 8
train_dataloader = torch.utils.data.DataLoader(train_data,batch_size,shuffle=True)
#val_dataloader = torch.utils.data.DataLoader(val_data,batch_size,shuffle=True)
sample_model = Sample(sigma,factor)
sample_model.to(device)
recon_model.to(device)

# %% optimization parameters
recon_optimizer = optim.Adam(recon_model.parameters(),lr=3e-4)
L1Loss = torch.nn.L1Loss()
step = 3e2 # sampling weight optimization step size
L2Loss = torch.nn.MSELoss()

# %% training
max_epochs = 50
#val_loss = torch.zeros(max_epochs)
for epoch in range(max_epochs):
    print("epoch:",epoch+1)
    step = step * 0.9
    batch_count = 0
    for train_batch in train_dataloader:
        
        batch_count = batch_count + 1
        sample_model.mask.requires_grad = True
        train_batch.to(device)
        
        gt = toIm(train_batch) # ground truth
        kspace_noise = sample_model(train_batch).to(device) # add noise
        
        # forward
        image_noise = fastmri.ifft2c(kspace_noise).to(device)
        image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1).to(device) 
        image_output = recon_model(image_input).to(device)
        image_recon = torch.cat((image_output[:,torch.arange(16),:,:].unsqueeze(4),image_output[:,torch.arange(16,32),:,:].unsqueeze(4)),4).to(device)
        recon = fastmri.rss(fastmri.complex_abs(image_recon), dim=1)
        loss = L2Loss(recon.to(device),gt.to(device))

        if batch_count%100 == 0:
            print("batch:",batch_count,"train loss:",loss.item())
        
        # backward
        loss.backward()
        
        # optimize mask
        with torch.no_grad():
            grad = sample_model.mask.grad
            temp = sample_model.mask.clone()
            sample_model.mask = temp - step * grad
        
        # optimize network
        recon_optimizer.step()
        recon_optimizer.zero_grad()

    torch.save(recon_model,"/project/jhaldar_118/jiayangw/OptSamp/model/opt_L2_model_sigma"+str(sigma))
    torch.save(sample_model.mask,"/project/jhaldar_118/jiayangw/OptSamp/model/opt_L2_mask_sigma"+str(sigma))

