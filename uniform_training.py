# %%
import pathlib
import torch
import torch.optim as optim
import fastmri
from fastmri.models import Unet
from fastmri.data import transforms, mri_data

# %% data loader
def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the kspace to tensor format
    kspace = transforms.to_tensor(kspace)
    return kspace

train_data = mri_data.SliceDataset(
    #root=pathlib.Path('/home/wjy/Project/fastmri_dataset/multicoil_test/T2/'),
    root = pathlib.Path('/project/jhaldar_118/fastMRI_dataset/Brain/MultiCoil/multicoil_test/'),
    transform=data_transform,
    challenge='multicoil'
)



# %% noise generator and transform to image
class Sample(torch.nn.Module): 

    def __init__(self,sigma,mask):
        super().__init__()
        self.mask = torch.sqrt(mask)
        self.sigma = sigma

    def forward(self,kspace):
        noise = sigma*torch.randn_like(kspace)
        kspace_noise = kspace + torch.div(noise,self.mask.unsqueeze(0).unsqueeze(3))  # need to reshape mask
        image = fastmri.ifft2c(kspace_noise)
        image = fastmri.complex_abs(image)
        image = fastmri.rss(image,dim=1).unsqueeze(1)
        gt = fastmri.ifft2c(kspace)
        gt = fastmri.complex_abs(gt)
        gt = fastmri.rss(gt,dim=1).unsqueeze(1)
        image = transforms.normalize(image,gt.mean(),gt.std(),1e-11)
        return image


class toImage(torch.nn.Module): 

    def __init__(self):
        super().__init__()

    def forward(self,kspace):
        image = fastmri.ifft2c(kspace)
        image = fastmri.complex_abs(image)
        image = fastmri.rss(image,dim=1).unsqueeze(1)
        image = transforms.normalize_instance(image,1e-11)
        return image[0]


# %% sampling
factor = 8
mask = torch.ones_like(train_data[0])
mask = factor*mask[0,:,:,0].squeeze() 
#mask.requires_grad = True
sigma = 5e-5
sample_model = Sample(sigma,mask)

toIm = toImage()

# %% unet loader
recon_model = Unet(
  in_chans = 1,
  out_chans = 1,
  chans = 32,
  num_pool_layers = 4,
  drop_prob = 0.0
)


# %% GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available() == False:
    batch_size = 1
    print("Let's use",device)
else:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    batch_size = torch.cuda.device_count()
    sample_model = torch.nn.DataParallel(sample_model)
    recon_model = torch.nn.DataParallel(recon_model)
    toIm = torch.nn.DataParallel(toIm)
train_dataloader = torch.utils.data.DataLoader(train_data,batch_size,shuffle=True)
sample_model.to(device)
recon_model.to(device)
toIm.to(device)

# %% optimizer
recon_optimizer = optim.RMSprop(recon_model.parameters(),lr=1e-3)

def NRMSE_loss(recon,ground_truth):
    return torch.norm(recon-ground_truth)/torch.norm(ground_truth)


# %% training
max_epochs = 10
for epoch in range(max_epochs):
    print("epoch:",epoch)
    batch_count = 0
    for train_batch in train_dataloader:
        batch_count = batch_count + 1
        #print("batch:",batch_count)
        train_batch.to(device)
        image_noise = sample_model(train_batch)
        recon = recon_model(image_noise)
        ground_truth = toIm(train_batch)
        loss = NRMSE_loss(recon,ground_truth)
        if batch_count%10 == 0:
            print("batch:",batch_count,"train loss:",loss.item(),"Original NRMSE:", NRMSE_loss(image_noise,ground_truth))
        loss.backward
        recon_optimizer.step()
        recon_optimizer.zero_grad()

    torch.save(recon_model,"./uniform_model")
# %% save model
#torch.save(recon_model,"./uniform_model")

