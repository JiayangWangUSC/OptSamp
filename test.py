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

# %% noise generator and transform to image
batch_size = 8

class Sample(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.mask = torch.ones(396)
        self.factor = factor
        self.sigma = sigma

    def forward(self,kspace):
        sample_mask = torch.sqrt(F.softmax(self.mask)*(self.factor-1)*396+1)
        torch.manual_seed(10)
        noise = self.sigma*torch.randn_like(kspace)
        kspace_noise = kspace + torch.div(noise,sample_mask.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(0).repeat(kspace.size(0),16,384,1,2))  # need to reshape mask        image = fastmri.ifft2c(kspace_noise)
        return kspace_noise

def toIm(kspace): 
    image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1)
    return image


# %% sampling
factor = 8
sigma = 0.3
sample_model = Sample(sigma,factor)

# %% image unet
sample_model = Sample(sigma,factor)
#mask = torch.load('/home/wjy/mask_image_unet_L1loss_noise0.3')
#sample_model.mask = mask
#Mask = F.softmax(mask)*(factor-1)*396+1
recon_model = torch.load('/home/wjy/opt_image_unet_L1loss_noise0.3',map_location=torch.device('cpu'))
# %%
kspace = test_data[3]
kspace = kspace.unsqueeze(0)
Im  = toIm(kspace)
support = torch.ge(Im,0.05*torch.max(Im))
with torch.no_grad():
    kspace_noise = sample_model(kspace)
    ImN = toIm(kspace_noise)
    image_noise = fastmri.ifft2c(kspace_noise)
    image_input = torch.cat((image_noise[:,:,:,:,0],image_noise[:,:,:,:,1]),1) 
    image_output = recon_model(image_input)
    image_recon = torch.cat((image_output[:,torch.arange(16),:,:].unsqueeze(4),image_output[:,torch.arange(16,32),:,:].unsqueeze(4)),4)
    recon = fastmri.rss(fastmri.complex_abs(image_recon), dim=1)
#Error = torch.abs(ImR-Im)
#plt.imshow(Error[0,:,:]/torch.max(Im)*10,cmap='hot')


#%% kspace unet
sample_model = Sample(sigma,factor)
mask = torch.load('/home/wjy/mask_kspace_unet_L1loss_noise0.3')
sample_model.mask = mask
Mask = F.softmax(mask)*(factor-1)*396+1
recon_model = torch.load('/home/wjy/opt_kspace_unet_L1loss_noise0.3',map_location=torch.device('cpu'))
# %%
kspace = test_data[1]
kspace = kspace.unsqueeze(0)
Im  = toIm(kspace)
support = torch.ge(Im,0.05*torch.max(Im))
with torch.no_grad():
    kspace_noise = sample_model(kspace)
    kspace_input = torch.cat((kspace_noise[:,:,:,:,0],kspace_noise[:,:,:,:,1]),1) 
    kspace_output = recon_model(kspace_input)
    kspace_recon = torch.cat((kspace_output[:,torch.arange(16),:,:].unsqueeze(4),kspace_output[:,torch.arange(16,32),:,:].unsqueeze(4)),4)
    recon = toIm(kspace_recon)


# %%
cmhot = plt.cm.get_cmap('hot')
Error = cmhot(np.array(Error.squeeze()/torch.max(Im)*10))
Error = np.uint8(Error*255)
Error = Image.fromarray(Error)
Error.save('/home/wjy/Project/OptSamp/result_local/NN_error_opt_L1_noise05.png')
# %%
cmhot = plt.cm.get_cmap('jet')
Mask = F.softmax(mask)*(factor-1)*396+1
Mask = cmhot(np.array(Mask.unsqueeze(0).repeat([384,1])/15))
Mask = np.uint8(Mask*255)
Mask = Image.fromarray(Mask)
Mask.save('/home/wjy/Project/OptSamp/result_local/kspaceunet_mask_L1_noise03.png')
# %%

ImR = ImR.squeeze().numpy()
import scipy.io
scipy.io.savemat('/home/wjy/Project/OptSamp/file/NN_uniform_recon_Noise2.mat', {"data":ImR})

#%%
N1 = 384
N2 = 396
# %% mask _extraction
def support_extraction(Batch):
    result = Batch.clone()
    N1 = Batch.size(2)
    N2 = Batch.size(3)
    for batch in range(Batch.size(0)):
        Im = Batch[batch,0,:,:].squeeze()
        support = torch.ge(Im,0.1*Im.max())
        out_mask = torch.zeros(N1,N2)

        for n1 in range(N1):
            if torch.sum(support[n1,:])==0:
                continue;
            head, tail, n2_head, n2_tail = N2,-1,-1, N2 
            while head>N2-1:
                n2_head += 1
                if support[n1,n2_head]==1:
                    head = n2_head
            while tail<0:
                n2_tail -= 1
                if support[n1,n2_tail]==1:
                    tail = n2_tail
            for n2 in range(head,tail+1):
                out_mask[n1,n2] += 1

        for n2 in range(N2):
            if torch.sum(support[:,n2])==0:
                continue;
            head, tail, n1_head, n1_tail = N1,-1,-1, N1 
            while head>N1-1:
                n1_head += 1
                if support[n1_head,n2]==1:
                    head = n1_head
            while tail<0:
                n1_tail -= 1
                if support[n1_tail,n2]==1:
                    tail = n1_tail
            for n1 in range(head,tail+1):
                out_mask[n1,n2] += 1
        out_mask = torch.ge(out_mask,2)

        inner_mask = torch.zeros(N1,N2)
        inv_support = torch.mul(~support,out_mask)

        for n1 in range(N1):
            if torch.sum(inv_support[n1,:])==0:
                continue;
            head, tail, n2_head, n2_tail = N2,-1,-1, N2 
            while head>N2-1:
                n2_head += 1
                if inv_support[n1,n2_head]==1:
                    head = n2_head
            while tail<0:
                n2_tail -= 1
                if inv_support[n1,n2_tail]==1:
                    tail = n2_tail
            for n2 in range(head,tail+1):
                inner_mask[n1,n2] += 1

        for n2 in range(N2):
            if torch.sum(inv_support[:,n2])==0:
                continue;
            head, tail, n1_head, n1_tail = N1,-1,-1, N1 
            while head>N1-1:
                n1_head += 1
                if inv_support[n1_head,n2]==1:
                    head = n1_head
            while tail<0:
                n1_tail -= 1
                if inv_support[n1_tail,n2]==1:
                    tail = n1_tail
            for n1 in range(head,tail+1):
                inner_mask[n1,n2] += 1
        inner_mask = torch.ge(inner_mask,2)
        result[batch,0,:,:] = inner_mask
    return result

support = support_extraction(Im)
# %%

D1 = torch.zeros(N1,N1)
D2 = torch.zeros(N2,N2)

for n1 in range(N1):
    D1[n1,n1] = 1
    if n1< N1-1:
        D1[n1,n1+1] = -1
    else:
        D1[n1,0] = -1

for n2 in range(N2):
    D2[n2,n2] = 1
    if n2< N2-1:
        D2[n2,n2+1] = -1
    else:
        D2[n1,0] = -1


def GradMap(Batch,support,D1,D2):
    gradmap = support.clone()
    N1 = Batch.size(2)
    N2 = Batch.size(3)
    for batch in range(Batch.size(0)):
        Im = Batch[batch,0,:,:].squeeze()
        sp = support[batch,0,:,:].squeeze()
        G1 = torch.matmul(D1,Im)
        G2 = torch.matmul(Im,D2)
        G = torch.sqrt(torch.mul(G1,G1)+torch.mul(G2,G2))
        G = torch.reshape(torch.mul(G,sp),(-1,))
        sorted,_ = torch.sort(G,descending=True)
        th = sorted[round(torch.sum(sp).item()*0.3)]
        gradmap[batch,0,:,:] = torch.reshape(torch.ge(G,th),(N1,N2))

    return gradmap

gradmap = GradMap(Im,support,D1,D2)
