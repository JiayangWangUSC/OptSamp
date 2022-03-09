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
from PIL import Image
import numpy as np

# %% data loader
def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the kspace to tensor format
    kspace = transforms.to_tensor(kspace)
    image = fastmri.ifft2c(kspace)
    image = image[:,torch.arange(191,575),:,:]
    kspace = fastmri.fft2c(image)/1e-4
    return kspace

test_data = mri_data.SliceDataset(
    root=pathlib.Path('/home/wjy/Project/fastmri_dataset/test/'),
    #root = pathlib.Path('/project/jhaldar_118/jiayangw/OptSamp/dataset/train/'),
    transform=data_transform,
    challenge='multicoil'
)

#val_data = mri_data.SliceDataset(
#    root=pathlib.Path('/home/wjy/Project/fastmri_dataset/multicoil_test/T2/'),
#    #root = pathlib.Path('/project/jhaldar_118/jiayangw/OptSamp/dataset/val/'),
#    transform=data_transform,
#    challenge='multicoil'
#)

# to image
def toIm(kspace): 
    image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1)
    return image


# %% parameters
factor = 8
batch_size = 8
sigma = 0.1
L1Loss = torch.nn.L1Loss()
# %% image unet uniform
class Sample(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.mask = 2*torch.rand(396)-1
        self.factor = factor
        self.sigma = sigma

    def forward(self,kspace):
        sample_mask = torch.sqrt(1 + F.softmax(self.mask)*(self.factor-1)*396)
        torch.manual_seed(20)
        noise = self.sigma*torch.randn_like(kspace)
        kspace_noise = kspace + torch.div(noise,sample_mask.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(0).repeat(kspace.size(0),16,384,1,2)) 
        return kspace_noise

sample_model = Sample(sigma,factor)
recon_model = torch.load('/home/wjy/Project/optsamp_models/uni_model_noise'+str(sigma),map_location=torch.device('cpu'))

# %% opt
sample_model = Sample(sigma,factor)
#mask = torch.load('/home/wjy/opt_mask_L1loss_noise0.3')
mask = torch.load('/home/wjy/Project/optsamp_models/opt_mask_noise'+str(sigma))
sample_model.mask = mask
sample_mask = 1 + F.softmax(mask)*(factor-1)*396
recon_model = torch.load('/home/wjy/Project/optsamp_models/opt_model_noise'+str(sigma),map_location=torch.device('cpu'))

# %% low frequency
class Sample(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.factor = factor
        self.sigma = sigma

    def forward(self,kspace):
        torch.manual_seed(10)
        noise = self.sigma*torch.randn_like(kspace)/math.sqrt(self.factor/0.8)
        support = torch.zeros(396)
        support[torch.arange(38,38+320)] = 1
        kspace_noise = torch.mul(kspace + noise, support.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(0).repeat(kspace.size(0),16,384,1,2))
        return kspace_noise

sample_model = Sample(sigma,factor)
recon_model = torch.load('/home/wjy/Project/optsamp_models/low_model_noise'+str(sigma),map_location=torch.device('cpu'))

# %%
kspace = test_data[0]
kspace = kspace.unsqueeze(0)
Im  = toIm(kspace)
with torch.no_grad():
    kspace_noise = sample_model(kspace)
    ImN = toIm(kspace_noise)
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1) 
    image_output = recon_model(image_input)
    image_recon = torch.cat((image_output[:,torch.arange(16),:,:].unsqueeze(4),image_output[:,torch.arange(16,32),:,:].unsqueeze(4)),4)
    recon = fastmri.rss(fastmri.complex_abs(image_recon), dim=1)

# %%
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

ssim_module = SSIM(data_range=255, size_average=True, channel=1)

print(torch.norm(Im-ImN)/torch.norm(Im))
print(ssim_module(Im.unsqueeze(0)/torch.max(Im)*255,recon.unsqueeze(0)/torch.max(Im)*255))
print(torch.norm(Im-recon)/torch.norm(Im))
print(torch.sum(torch.abs(Im-recon))/torch.sum(torch.abs(Im)))


# %%
save_image(recon.squeeze()/torch.max(Im)*2,'/home/wjy/Project/optsamp_result/slice0_noise1_opt.png')
save_image(torch.abs(recon-Im).squeeze()/torch.max(Im)*15,'/home/wjy/Project/optsamp_result/slice0_noise1_opt_error.png')

# %%
cmhot = plt.cm.get_cmap('jet')
Mask = F.softmax(mask)*(factor-1)*396+1
Mask = cmhot(np.array(Mask.unsqueeze(0).repeat([384,1])/18))
Mask = np.uint8(Mask*255)
Mask = Image.fromarray(Mask)
Mask.save('/home/wjy/Project/OptSamp/result_local/mask_rand_noise03.png')

# %% plot mask
fig = plt.figure()

low_mask = torch.zeros(396)
low_mask[torch.arange(38,38+320)] = 10
uni_mask = 8*torch.ones(396)
mask1 = torch.load('/home/wjy/Project/optsamp_models/opt_mask_noise0.1')
mask1 = 1 + F.softmax(mask1)*(factor-1)*396
mask3 = torch.load('/home/wjy/Project/optsamp_models/opt_mask_noise0.3')
mask3 = 1 + F.softmax(mask3)*(factor-1)*396
mask4 = torch.load('/home/wjy/Project/optsamp_models/opt_mask_noise0.4')
mask4 = 1 + F.softmax(mask4)*(factor-1)*396
mask6 = torch.load('/home/wjy/Project/optsamp_models/opt_mask_noise0.6')
mask6 = 1 + F.softmax(mask6)*(factor-1)*396

plt.plot(uni_mask,label='uniform')
plt.plot(low_mask,label='low frequency')
plt.plot(mask1,label='Noise 1')
plt.plot(mask3,label='Noise 3')
plt.plot(mask4,label='Noise 4')
plt.plot(mask6,label='Noise 6')

plt.legend()
plt.show()

fig.savefig('/home/wjy/Project/optsamp_result/mask.png')
# %%
