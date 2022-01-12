
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
# %%
class NormUnet(nn.Module):
    """
    Normalized U-Net model.
    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.
    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, h, w, comp)
    
    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        return x.view(b * c, 1, h, w, comp), b

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def forward(
        self,
        acs_kspace: torch.Tensor
    ) -> torch.Tensor:

        # convert to image space
        images, batches = self.chans_to_batch_dim(fastmri.ifft2c(acs_kspace))
        # estimate sensitivities
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        )


class VarNet(nn.Module):
    """
    A full variational network model.
    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()

        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        acs_kspace:torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        sens_maps = self.sens_net(acs_kspace)
        kspace_pred = torch.div(masked_kspace.clone(),mask)

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)


class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.
    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.complex_mul(
            fastmri.ifft2c(x), fastmri.complex_conj(sens_maps)
        ).sum(dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        soft_dc = torch.mul(mask,(torch.mul(mask, current_kspace) - ref_kspace)) * self.dc_weight
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps
        )

        return current_kspace - soft_dc + model_term

#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# %% data loader
def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the kspace to tensor format
    kspace = transforms.to_tensor(kspace)
    image = fastmri.ifft2c(kspace)
    image = image[:,torch.arange(191,575),:,:]
    kspace = fastmri.fft2c(image)/1e-4
    return kspace

train_data = mri_data.SliceDataset(
    #root=pathlib.Path('/home/wjy/Project/fastmri_dataset/test/'),
    root = pathlib.Path('/project/jhaldar_118/jiayangw/OptSamp/dataset/train/'),
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
        self.mask = factor*torch.ones(396)
        #self.mask = torch.rand(396)
        #self.mask = self.mask/torch.mean(self.mask)*factor
        self.sigma = sigma

    def forward(self,kspace):
        noise = self.sigma*torch.randn_like(kspace)
        kspace_noise = noise + torch.mul(kspace,torch.sqrt(self.mask).unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(0).repeat(kspace.size(0),16,384,1,2))  # need to reshape mask        image = fastmri.ifft2c(kspace_noise)
        return kspace_noise

def toIm(kspace): 
    image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1)
    return image
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

# %% sampling
factor = 8
sigma = 0.3
print("noise level:", sigma)
sample_model = Sample(sigma,factor)


# %% unet loader
#recon_model = Unet(
#  in_chans = 1,
#  out_chans = 1,
#  chans = 32,
#  num_pool_layers = 4,
#  drop_prob = 0.0
#)

recon_model = VarNet(
    num_cascades = 4,
    sens_chans = 8,
    sens_pools = 4,
    chans = 16,
    pools = 4
)


#recon_model = torch.load('/project/jhaldar_118/jiayangw/OptSamp/uniform_model_selfloss_noise'+str(sigma))

# %% GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataloader = torch.utils.data.DataLoader(train_data,batch_size,shuffle=True)
#val_dataloader = torch.utils.data.DataLoader(val_data,batch_size,shuffle=True)

sample_model.to(device)
recon_model.to(device)


# %% gradient computation
N1 = 384
N2 = 396
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
    N1 = Batch.size(1)
    N2 = Batch.size(2)
    for batch in range(Batch.size(0)):
        Im = Batch[batch,:,:].squeeze()
        sp = support[batch,:,:].squeeze()
        G1 = torch.matmul(D1,Im)
        G2 = torch.matmul(Im,D2)
        G = torch.sqrt(torch.mul(G1,G1)+torch.mul(G2,G2))
        G = torch.reshape(torch.mul(G,sp),(-1,))
        sorted,_ = torch.sort(G,descending=True)
        th = sorted[round(torch.sum(sp).item()*0.3)]
        gradmap[batch,:,:] = torch.reshape(torch.ge(G,th),(N1,N2))

    return gradmap

# %% optimizer
recon_optimizer = optim.Adam(recon_model.parameters(),lr=1e-3)
#Loss = torch.nn.MSELoss()
L1Loss = torch.nn.L1Loss()
L2Loss = torch.nn.MSELoss()
beta = 1e-3
#ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=1)

# %% training
step = 1e-1
max_epochs = 10
#val_loss = torch.zeros(max_epochs)
for epoch in range(max_epochs):
    print("epoch:",epoch+1)
    step = step*0.9
    batch_count = 0
    for train_batch in train_dataloader:
        batch_count = batch_count + 1
        sample_model.mask.requires_grad = True
        train_batch.to(device)

        gt = toIm(train_batch)
        support = torch.ge(gt,0.06*torch.max(gt))
        gradmap = GradMap(gt,support,D1,D2)
        
        acs_kspace = torch.zeros_like(train_batch)
        acs_kspace[:,:,:,torch.arange(186,210),:] = 1
        acs_kspace = torch.mul(acs_kspace,train_batch).to(device)

        kspace_noise = sample_model(train_batch).to(device)
        mask = torch.sqrt(sample_model.mask).unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(0).repeat(train_batch.size(0),16,384,1,2).to(device)
        recon = recon_model(kspace_noise, acs_kspace, mask)
        loss = L2Loss(torch.mul(recon.to(device),support.to(device)),torch.mul(gt.to(device),support.to(device))) + beta*L1Loss(torch.mul(recon.to(device),gradmap.to(device)),torch.mul(gt.to(device),gradmap.to(device)))
        #loss = L1Loss(torch.mul(recon.to(device),support.to(device)),torch.mul(gt.to(device),support.to(device)))
        #loss = 1- ms_ssim_module(recon*25,recon*25)

        if batch_count%100 == 0:
            print("batch:",batch_count,"train loss:",loss.item())
        
        loss.backward()

        with torch.no_grad():
            grad = sample_model.mask.grad
            grad = (grad - torch.mean(grad))/torch.std(grad)
            temp = sample_model.mask.clone()
            temp = temp - step * grad
            for p in range(10):
                temp = torch.relu(temp-1)+1
                temp = temp - torch.mean(temp) + factor
            sample_model.mask = torch.relu(temp-1)+1

        recon_optimizer.step()
        recon_optimizer.zero_grad()

  #  with torch.no_grad():
  #      loss = 0
  #     orig_loss = 0
  #      for val_batch in val_dataloader:
  #          gt = toIm(val_batch)
  #          support = support_extraction(gt)
  #          gradmap = GradMap(gt,support,D1,D2)
  #          gt.to(device)
  #          support.to(device)
  #          gradmap.to(device)
  #          train_batch.to(device)

  #          image_noise = sample_model(val_batch).to(device)
  #          recon = recon_model(image_noise).to(device)
  #
  #          ground_truth = torch.mul(ground_truth,support).to(device)
  #          
  #          loss += L1Loss(recon,ground_truth)
  #          orig_loss += L1Loss(image_noise,ground_truth)

#        val_loss[epoch] = loss/len(val_dataloader)
#        print("epoch:",epoch+1,"validation Loss:",val_loss[epoch])

   # torch.save(val_loss,"./uniform_model_val_loss_noise"+str(sigma))
    torch.save(recon_model,"./opt_varnet_L12loss_noise"+str(sigma))
    torch.save(sample_model.mask,"./mask_varnet_L12loss_noise"+str(sigma))
# %%
