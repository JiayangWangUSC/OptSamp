# %%
import pathlib
import torch
import numpy as np
import math
import torch.optim as optim
import fastmri
from fastmri.models import Unet
from fastmri.data import transforms, mri_data
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image
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

# %% noise generator and transform to image
batch_size = 8

class Sample(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.mask = factor*torch.ones(396)
        self.sigma = sigma

    def forward(self,kspace):

        noise = self.sigma*torch.randn_like(kspace)
        kspace_noise = kspace + torch.div(noise,torch.sqrt(self.mask).unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(0).repeat(kspace.size(0),16,384,1,2))  # need to reshape mask        image = fastmri.ifft2c(kspace_noise)
        image = fastmri.ifft2c(kspace_noise)
        image = torch.sqrt(torch.sum(torch.sum(torch.mul(image,image),4),1)).unsqueeze(1)
        #image = transforms.normalize(image,glob_mean,glob_std,1e-11)
        return image

class toImage(torch.nn.Module): 

    def __init__(self):
        super().__init__()

    def forward(self,kspace):
        image = fastmri.ifft2c(kspace)
        image = torch.sqrt(torch.sum(torch.sum(torch.mul(image,image),4),1)).unsqueeze(1)
        #image = transforms.normalize(image,glob_mean,glob_std,1e-11)
        return image


# %% sampling
factor = 8
sigma = 0.5
sample_model = Sample(sigma,factor)

toIm = toImage()


# %% load uniform-unet model
#val_uniform_loss = torch.load('/home/wjy/unet_model_val_loss')
mask = torch.load('/home/wjy/unet_mask_L1_noise0.5')
sample_model.mask = mask
model = torch.load('/home/wjy/unet_model_L1_noise0.5',map_location=torch.device('cpu'))
# %%
#plt.plot(val_loss)
# %%

kspace = test_data[2]
kspace = kspace.unsqueeze(0)
Im  = toIm(kspace)
#plt.imshow(Im[0,0,:,:],cmap='gray')
ImN = sample_model(kspace)
#plt.imshow(ImN[0,0,:,:],cmap='gray')
with torch.no_grad():
    ImR = model(ImN)

support = torch.ge(Im,0.06*Im.max())
ImR = torch.mul(ImR,support)
Im = torch.mul(Im,support)
Error = torch.abs(ImR-Im)
plt.imshow(Error[0,0,:,:]/torch.max(Im)*10,cmap='hot')
# %%
cmhot = plt.cm.get_cmap('hot')
Error = cmhot(np.array(Error.squeeze()/torch.max(Im)*10))
Error = np.uint8(Error*255)
Error = Image.fromarray(Error)
Error.save('/home/wjy/Project/OptSamp/result_local/NN_error_opt_L1_noise03.png')
# %%
cmhot = plt.cm.get_cmap('jet')
Mask = cmhot(np.array(mask.unsqueeze(0).repeat([384,1])/6-1))
Mask = np.uint8(Mask*255)
Mask = Image.fromarray(Mask)
Mask.save('/home/wjy/Project/OptSamp/result_local/NN_mask_noise03.png')
# %%

ImR = ImR.squeeze().numpy()
import scipy.io
scipy.io.savemat('/home/wjy/Project/OptSamp/file/NN_uniform_recon_Noise2.mat', {"data":ImR})
# %%
