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
#mask = torch.load('/home/wjy/unet_mask_selfloss_noise0.5')
#sample_model.mask = mask
model = torch.load('/home/wjy/uniform_model_selfloss_noise0.5',map_location=torch.device('cpu'))

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
Error.save('/home/wjy/Project/OptSamp/result_local/NN_error_opt_L1_noise05.png')
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
