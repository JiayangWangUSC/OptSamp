# %%
import pathlib
import torch
import fastmri
from fastmri.models import Unet
from fastmri.data import transforms, mri_data

# %% data loader
def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the kspace to tensor format
    kspace = transforms.to_tensor(kspace)
    return kspace

train_data = mri_data.SliceDataset(
    root=pathlib.Path(
      '/home/wjy/Project/fastmri_dataset/multicoil_test'
    ),
    transform=data_transform,
    challenge='multicoil'
)



# %% noise generator and transform to image
class Sample(torch.nn.Module): 

    def __init__(self,sigma,mask):
        super().__init__()
        self.mask = mask
        self.sigma = sigma

    def forward(self,kspace):
        noise = sigma*torch.randn_like(kspace)
        kspace_noise = kspace + torch.div(noise,self.mask)  # need to reshape mask
        image = fastmri.ifft2c(kspace_noise)
        image = fastmri.complex_abs(image)
        image = transforms.normalize_instance(image,1e-11)
        return image

# %% GPU set up
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("Let's use", torch.cuda.device_count(), "GPUs!")

batch_size = torch.cuda.device_count()
train_dataloader = torch.utils.data.DataLoader(train_data,batch_size,shuffle=True)
# %% unet loader
recon_model = Unet(
  in_chans = 1
  out_chans = 1
  chans = 32
  num_pool_layer = 4
  drop_prob = 0.0
)
recon_model = torch.nn.DataParallel(recon_model)
recon_model.to(device)
# %% sampling
mask = torch.ones() 
sigma = 1
sample_model = Sample(sigma,mask)
sample_model = torch.nn.DataParallel(sample_model)
sample_model.to(device)

# %% training
max_epochs = 10
for epoch in range(max_epochs):
    for train_batch in train_dataloader:
      train_batch.to(device)
