# %%
import pathlib
import torch
import torch.optim as optim
import fastmri
from fastmri.models import Unet
from fastmri.data import transforms, mri_data
import matplotlib.pyplot as plt
from torchvision.utils import save_image
# %% data loader
def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the kspace to tensor format
    kspace = transforms.to_tensor(kspace)
    return kspace

test_data = mri_data.SliceDataset(
    root=pathlib.Path('/home/wjy/Project/fastmri_dataset/train/'),
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
glob_mean = 0
glob_std = 5e-5
batch_size = 8

class Sample(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.mask = torch.ones_like(test_data[0])
        self.mask = factor*self.mask[0,:,:,0].squeeze() 
        self.sigma = sigma

    def forward(self,kspace):
        noise = self.sigma*torch.randn_like(kspace)
        kspace_noise = kspace + torch.div(noise,torch.sqrt(self.mask).unsqueeze(0).unsqueeze(3).unsqueeze(0).repeat(kspace.size(0),16,1,1,2))  # need to reshape mask        image = fastmri.ifft2c(kspace_noise)
        image = fastmri.ifft2c(kspace_noise)
        image = fastmri.complex_abs(image)
        image = fastmri.rss(image,dim=1).unsqueeze(1)
        image = transforms.normalize(image,glob_mean,glob_std,1e-11)
        return image


class toImage(torch.nn.Module): 

    def __init__(self):
        super().__init__()

    def forward(self,kspace):
        image = fastmri.ifft2c(kspace)
        image = fastmri.complex_abs(image)
        image = fastmri.rss(image,dim=1).unsqueeze(1)
        image = transforms.normalize(image,glob_mean,glob_std,1e-11)
        return image


# %% sampling
factor = 8
sigma = 5e-5
sample_model = Sample(sigma,factor)

toIm = toImage()


# %% load uniform-unet model
val_uniform_loss = torch.load('/home/wjy/unet_model_val_loss')
model = torch.load('/home/wjy/uniform_model',map_location=torch.device('cpu'))
# %%
plt.plot(val_loss)
# %%
kspace = test_data[0]
print(kspace.size(0))
kspace = kspace.unsqueeze(0)
Im  = toIm(kspace).squeeze()
#plt.imshow(Im[0,0,:,:],cmap='gray')
ImN = sample_model(kspace).squeeze()
#plt.imshow(ImN[0,0,:,:],cmap='gray')
#with torch.no_grad():

#    ImR = model(ImN)
#plt.imshow(ImR[0,0,:,:],cmap='gray')
# %%
mask = torch.load('/home/wjy/unet_mask')
val_unet_loss = torch.load('/home/wjy/unet_model_val_loss')