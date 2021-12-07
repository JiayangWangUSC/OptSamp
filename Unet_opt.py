# %%
import pathlib
import torch
import torch.optim as optim
import fastmri
from fastmri.models import Unet
from fastmri.data import transforms, mri_data
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# %% data loader
def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the kspace to tensor format
    kspace = transforms.to_tensor(kspace)
    image = fastmri.ifft2c(kspace)
    image = image[:,torch.arange(191,575),:,:]
    kspace = fastmri.fft2c(image)/1e-4
    return kspace

train_data = mri_data.SliceDataset(
    #root=pathlib.Path('/home/wjy/Project/fastmri_dataset/multicoil_test/T2/'),
    root = pathlib.Path('/project/jhaldar_118/jiayangw/OptSamp/dataset/train/'),
    transform=data_transform,
    challenge='multicoil'
)

val_data = mri_data.SliceDataset(
    #root=pathlib.Path('/home/wjy/Project/fastmri_dataset/multicoil_test/T2/'),
    root = pathlib.Path('/project/jhaldar_118/jiayangw/OptSamp/dataset/val/'),
    transform=data_transform,
    challenge='multicoil'
)

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

#%%
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

# %%
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
# %% sampling
factor = 8
sigma = 0.5
print("noise level:", sigma)
sample_model = Sample(sigma,factor)
#mask = torch.load('/project/jhaldar_118/jiayangw/OptSamp/unet_mask_L1_noise0.3')
#sample_model.mask = mask
toIm = toImage()

# %% unet loader
recon_model = Unet(
  in_chans = 1,
  out_chans = 1,
  chans = 32,
  num_pool_layers = 4,
  drop_prob = 0.0
)

#recon_model = torch.load('/project/jhaldar_118/jiayangw/OptSamp/unet_model_L1_noise0.3')

# %% 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 8

train_dataloader = torch.utils.data.DataLoader(train_data,batch_size,shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_data,batch_size,shuffle=True)

sample_model.to(device)
recon_model.to(device)
toIm.to(device)

# %% optimizer
recon_optimizer = optim.RMSprop(recon_model.parameters(),lr=1e-3)
#Loss = torch.nn.MSELoss()
L1Loss = torch.nn.L1Loss()
L2Loss = torch.nn.MSELoss()
#ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=1)
# %% training
step = 1e-1
max_epochs = 5
val_loss = torch.zeros(max_epochs)
for epoch in range(max_epochs):
    print("epoch:",epoch+1)
    step = step * 0.9
    batch_count = 0
    for train_batch in train_dataloader:
        batch_count = batch_count + 1
        sample_model.mask.requires_grad = True

        gt = toIm(train_batch)
        support = support_extraction(gt)
        gradmap = GradMap(gt,support,D1,D2)
        gt.to(device)
        support.to(device)
        gradmap.to(device)
        train_batch.to(device)
            
        image_noise = sample_model(train_batch).to(device)
        recon = recon_model(image_noise).to(device)
        loss = L2Loss(torch.mul(recon,support),torch.mul(gt,support)) + L1Loss(torch.mul(recon,gradmap),torch.mul(gt,gradmap))


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
            #print(torch.min(sample_model.mask))

        sample_model.mask.grad = None
        recon_optimizer.step()
        recon_optimizer.zero_grad(set_to_none=False)

  #  with torch.no_grad():
  #      loss = 0
  #      orig_loss = 0
  #      for val_batch in val_dataloader:
  #          val_batch.to(device)

  #           image_noise = sample_model(val_batch).to(device)
  #          recon = recon_model(image_noise).to(device)
  #          ground_truth = toIm(val_batch).to(device)
  #          support = torch.ge(ground_truth,0.06*ground_truth.max()).to(device)
  #          recon = torch.mul(recon,support).to(device)
  #          ground_truth = torch.mul(ground_truth,support).to(device)

  #          loss += L1Loss(recon,ground_truth)
  #          orig_loss += L1Loss(image_noise,ground_truth)

  #      val_loss[epoch] = loss/len(val_dataloader)
  #      print("epoch:",epoch+1,"validation MSE:",val_loss[epoch],"original MSE:",orig_loss/len(val_dataloader))

   # torch.save(val_loss,"./unet_model_val_loss_noise"+str(sigma))
    torch.save(recon_model,"./unet_model_selfloss_noise"+str(sigma))
    torch.save(sample_model.mask,"./unet_mask_selfloss_noise"+str(sigma))
