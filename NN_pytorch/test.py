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
    kspace = fastmri.fft2c(image)/5e-5
    return kspace

test_data = mri_data.SliceDataset(
    root=pathlib.Path('/home/wjy/Project/fastmri_dataset/test_brain/'),
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
sigma = 0.4
L1Loss = torch.nn.L1Loss()

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
ssim_module = SSIM(data_range=255, size_average=True, channel=1)

ssim_uni, ssim_opt, ssim_low = 0, 0, 0
mse_uni, mse_opt, mse_low = 0, 0, 0 

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

sample_uni = Sample(sigma,factor)
recon_uni = torch.load('/home/wjy/Project/optsamp_models/uni_model_noise'+str(sigma),map_location=torch.device('cpu'))

# %% opt
class Sample(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.mask = 2*torch.rand(396)-1
        self.factor = factor
        self.sigma = sigma

    def forward(self,kspace):
        sample_mask = torch.sqrt(1 + F.softmax(self.mask)*(self.factor-1)*396)
        torch.manual_seed(69)
        noise = self.sigma*torch.randn_like(kspace)
        kspace_noise = kspace + torch.div(noise,sample_mask.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(0).repeat(kspace.size(0),16,384,1,2)) 
        return kspace_noise

sample_opt = Sample(sigma,factor)
#mask = torch.load('/home/wjy/opt_mask_L1loss_noise0.3')
mask = torch.load('/home/wjy/Project/optsamp_models/opt_mask_noise'+str(sigma))
sample_opt.mask = mask
sample_mask = 1 + F.softmax(mask)*(factor-1)*396
recon_opt = torch.load('/home/wjy/Project/optsamp_models/opt_model_noise'+str(sigma),map_location=torch.device('cpu'))

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

sample_low80 = Sample(sigma,factor)
recon_low80 = torch.load('/home/wjy/Project/optsamp_models/low_model_noise'+str(sigma),map_location=torch.device('cpu'))

# %%
for slice in range(33,34):
    kspace = test_data[slice]
    kspace = kspace.unsqueeze(0)
    Im  = toIm(kspace)

    #save_image(Im.squeeze()/5,'/home/wjy/Project/optsamp_result/groundtruth/slice'+str(slice)+'.png')

    with torch.no_grad():
        # uniform sampling
        kspace_noise = sample_uni(kspace)
        ImN_uni = toIm(kspace_noise)
        save_image(ImN_uni.squeeze()/5,'/home/wjy/Project/optsamp_result/Unet/noise_slice'+str(slice)+'_noise'+str(int(10*sigma))+'_uni.png')
        print(torch.norm(Im-ImN_uni)/torch.norm(Im))
        image_noise = fastmri.ifft2c(kspace_noise)
        image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1) 
        image_output = recon_uni(image_input)
        image_recon = torch.cat((image_output[:,torch.arange(16),:,:].unsqueeze(4),image_output[:,torch.arange(16,32),:,:].unsqueeze(4)),4)
        recon_uni = fastmri.rss(fastmri.complex_abs(image_recon), dim=1)
        save_image(recon_uni.squeeze()/5,'/home/wjy/Project/optsamp_result/Unet/slice'+str(slice)+'_noise'+str(int(10*sigma))+'_uni.png')
        save_image(torch.abs(recon_uni-Im).squeeze()*2,'/home/wjy/Project/optsamp_result/Unet/error_slice'+str(slice)+'_noise'+str(int(10*sigma))+'_uni.png')
        ssim_uni = ssim_module(Im.unsqueeze(0)/5*255,recon_uni.unsqueeze(0)/5*255)
        mse_uni = torch.norm(Im-recon_uni)/torch.norm(Im)


        # optimized sampling
        kspace_noise = sample_opt(kspace)
        ImN_opt = toIm(kspace_noise)
        save_image(ImN_opt.squeeze()/5,'/home/wjy/Project/optsamp_result/Unet/noise_slice'+str(slice)+'_noise'+str(int(10*sigma))+'_opt.png')
        print(torch.norm(Im-ImN_opt)/torch.norm(Im))
        image_noise = fastmri.ifft2c(kspace_noise)
        image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1) 
        image_output = recon_opt(image_input)
        image_recon = torch.cat((image_output[:,torch.arange(16),:,:].unsqueeze(4),image_output[:,torch.arange(16,32),:,:].unsqueeze(4)),4)
        recon_opt = fastmri.rss(fastmri.complex_abs(image_recon), dim=1)
        save_image(recon_opt.squeeze()/5,'/home/wjy/Project/optsamp_result/Unet/slice'+str(slice)+'_noise'+str(int(10*sigma))+'_opt.png')
        save_image(torch.abs(recon_opt-Im).squeeze()*2,'/home/wjy/Project/optsamp_result/Unet/error_slice'+str(slice)+'_noise'+str(int(10*sigma))+'_opt.png')
        ssim_opt = ssim_module(Im.unsqueeze(0)/5*255,recon_opt.unsqueeze(0)/5*255)
        mse_opt = torch.norm(Im-recon_opt)/torch.norm(Im)

        # 80% low frequency sampling
        kspace_noise = sample_low80(kspace)
        ImN_low = toIm(kspace_noise)
        save_image(ImN_low.squeeze()/5,'/home/wjy/Project/optsamp_result/Unet/noise_slice'+str(slice)+'_noise'+str(int(10*sigma))+'_low.png')
        print(torch.norm(Im-ImN_low)/torch.norm(Im))
        image_noise = fastmri.ifft2c(kspace_noise)
        image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1) 
        image_output = recon_low80(image_input)
        image_recon = torch.cat((image_output[:,torch.arange(16),:,:].unsqueeze(4),image_output[:,torch.arange(16,32),:,:].unsqueeze(4)),4)
        recon_low = fastmri.rss(fastmri.complex_abs(image_recon), dim=1)
        save_image(recon_low.squeeze()/5,'/home/wjy/Project/optsamp_result/Unet/slice'+str(slice)+'_noise'+str(int(10*sigma))+'_low.png')
        save_image(torch.abs(recon_low-Im).squeeze()*2,'/home/wjy/Project/optsamp_result/Unet/error_slice'+str(slice)+'_noise'+str(int(10*sigma))+'_low.png')
        ssim_low = ssim_module(Im.unsqueeze(0)/5*255,recon_low.unsqueeze(0)/5*255)
        mse_low = torch.norm(Im-recon_low)/torch.norm(Im)


# %% patches

color_scale = 3

left = 110
right = 140 
up = 200
bottom = 230

# slice 33: 100, 140, 190, 230
# slice 64: 210, 250, 210, 250
# slice 114: 100,140,180,220
# slice 160: 140, 180, 180, 220
# slice 128

patch = Im
patch = patch[:,torch.arange(up,bottom),:] # slice160: (190,220) slice114: (180,210) 
patch = patch[:,:,torch.arange(left,right)] # slice160: (145,175) slice114: (110,140)
patch = F.interpolate(patch.unsqueeze(0),size=[256,256],mode='nearest')
save_image(patch.squeeze()/color_scale,'/home/wjy/Project/optsamp_result/Unet/patch1_slice'+str(slice)+'_gt.png')
patch = ImN_uni
patch = patch[:,torch.arange(up,bottom),:]
patch = patch[:,:,torch.arange(left,right)]
patch = F.interpolate(patch.unsqueeze(0),size=[256,256],mode='nearest')
save_image(patch.squeeze()/color_scale,'/home/wjy/Project/optsamp_result/Unet/patch1_slice'+str(slice)+'_noise'+str(int(10*sigma))+'_noiseuni.png')

patch = ImN_low
patch = patch[:,torch.arange(up,bottom),:]
patch = patch[:,:,torch.arange(left,right)]
patch = F.interpolate(patch.unsqueeze(0),size=[256,256],mode='nearest')
save_image(patch.squeeze()/color_scale,'/home/wjy/Project/optsamp_result/Unet/patch1_slice'+str(slice)+'_noise'+str(int(10*sigma))+'_noiselow.png')

patch = ImN_opt
patch = patch[:,torch.arange(up,bottom),:]
patch = patch[:,:,torch.arange(left,right)]
patch = F.interpolate(patch.unsqueeze(0),size=[256,256],mode='nearest')
save_image(patch.squeeze()/color_scale,'/home/wjy/Project/optsamp_result/Unet/patch1_slice'+str(slice)+'_noise'+str(int(10*sigma))+'_noiseopt.png')

patch = recon_uni
patch = patch[:,torch.arange(up,bottom),:]
patch = patch[:,:,torch.arange(left,right)]
patch = F.interpolate(patch.unsqueeze(0),size=[256,256],mode='nearest')
save_image(patch.squeeze()/color_scale,'/home/wjy/Project/optsamp_result/Unet/patch1_slice'+str(slice)+'_noise'+str(int(10*sigma))+'_uni.png')

patch = recon_low
patch = patch[:,torch.arange(up,bottom),:]
patch = patch[:,:,torch.arange(left,right)]
patch = F.interpolate(patch.unsqueeze(0),size=[256,256],mode='nearest')
save_image(patch.squeeze()/color_scale,'/home/wjy/Project/optsamp_result/Unet/patch1_slice'+str(slice)+'_noise'+str(int(10*sigma))+'_low.png')

patch = recon_opt
patch = patch[:,torch.arange(up,bottom),:]
patch = patch[:,:,torch.arange(left,right)]
patch = F.interpolate(patch.unsqueeze(0),size=[256,256],mode='nearest')
save_image(patch.squeeze()/color_scale,'/home/wjy/Project/optsamp_result/Unet/patch1_slice'+str(slice)+'_noise'+str(int(10*sigma))+'_opt.png')

# %%
print(torch.norm(Im-ImN)/torch.norm(Im))
print(ssim_module(Im.unsqueeze(0)/torch.max(Im)*255,recon.unsqueeze(0)/torch.max(Im)*255))
print(torch.norm(Im-recon)/torch.norm(Im))
print(torch.sum(torch.abs(Im-recon))/torch.sum(torch.abs(Im)))

# %%
save_image(recon.squeeze()/torch.max(Im)*2,'/home/wjy/Project/optsamp_result/slice0_noise3_opt.png')
save_image(torch.abs(recon-Im).squeeze()/torch.max(Im)*20,'/home/wjy/Project/optsamp_result/slice0_noise3_opt_error.png')

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
mask2 = torch.load('/home/wjy/Project/optsamp_models/opt_mask_noise0.2')
mask2 = 1 + F.softmax(mask2)*(factor-1)*396
mask4 = torch.load('/home/wjy/Project/optsamp_models/opt_mask_noise0.4')
mask4 = 1 + F.softmax(mask4)*(factor-1)*396
mask6 = torch.load('/home/wjy/Project/optsamp_models/opt_mask_noise0.6')
mask6 = 1 + F.softmax(mask6)*(factor-1)*396
mask8 = torch.load('/home/wjy/Project/optsamp_models/opt_mask_noise0.8')
mask8 = 1 + F.softmax(mask8)*(factor-1)*396

plt.plot(uni_mask,label='uniform')
plt.plot(low_mask,label='low frequency')
plt.plot(mask2,label='SNR=2.00')
plt.plot(mask4,label='SNR=1.00')
plt.plot(mask6,label='SNR=0.67')
plt.plot(mask8,label='SNR=0.50')

fig.patch.set_facecolor('black')

plt.legend()
plt.show()



fig.savefig('/home/wjy/Project/optsamp_result/Unet/mask.png')
# %%

uni_mask = uni_mask.cpu().detach().numpy()
low_mask = low_mask.cpu().detach().numpy()
mask2 = mask2.cpu().detach().numpy()
mask4 = mask4.cpu().detach().numpy()
mask6 = mask6.cpu().detach().numpy()
mask8 = mask8.cpu().detach().numpy()

import scipy.io

mask = {'uni':uni_mask, 'low':low_mask, 'mask2': mask2, 'mask4':mask4, 'mask6':mask6, 'mask8':mask8}
scipy.io.savemat('mask',mask)


# %%
cmhot = plt.cm.get_cmap('magma')
Error = torch.abs(recon_low-Im).squeeze()
error = Error + 6*(F.relu(Error,0.3)) + 2*(F.relu(Error,0.2))
Error = cmhot(np.array(error)/5)
Error = np.uint8(Error*255)
Error = Image.fromarray(Error)
Error.save('/home/wjy/Project/optsamp_result/Unet/error_slice'+str(slice)+'_noise'+str(int(10*sigma))+'_low.png')
# %%
error1 = torch.abs(recon_low-Im).squeeze().cpu().detach().numpy()
error2 = torch.abs(recon_opt-Im).squeeze().cpu().detach().numpy()
error3 = torch.abs(recon_uni-Im).squeeze().cpu().detach().numpy()
error = {'error_low':error1, 'error_opt':error2, 'error_uni':error3}
scipy.io.savemat('errormaps', error)
# %%
