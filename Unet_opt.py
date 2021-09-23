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
class Sample(torch.nn.Module): 

    def __init__(self,sigma,factor):
        super().__init__()
        self.mask = torch.ones_like(train_data[0])
        self.mask = factor*self.mask[0,:,:,0].squeeze() 
        self.sigma = sigma

    def forward(self,kspace):
        noise = self.sigma*torch.randn_like(kspace)
        kspace_noise = kspace + torch.div(noise,torch.sqrt(self.mask.unsqueeze(0).unsqueeze(3)))  # need to reshape mask
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
sigma = 5e-5
sample_model = Sample(sigma,factor)

toIm = toImage()

# %% unet loader
recon_model = Unet(
  in_chans = 1,
  out_chans = 1,
  chans = 32,
  num_pool_layers = 4,
  drop_prob = 0.0
)

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

def NRMSE_loss(recon,ground_truth):
    return torch.norm(recon-ground_truth)/torch.norm(ground_truth)

# %% training
step = 1
max_epochs = 10
val_loss = torch.zeros(max_epochs)
for epoch in range(max_epochs):
    print("epoch:",epoch+1)

    batch_count = 0
    for train_batch in train_dataloader:
        batch_count = batch_count + 1
        
        train_batch.to(device)
        
        sample_model.mask.requires_grad = True
        
        image_noise = sample_model(train_batch)
        recon = recon_model(image_noise.to(device))
        ground_truth = toIm(train_batch)

        loss = NRMSE_loss(recon.to(device),ground_truth.to(device))
        if batch_count%100 == 0:
            print("batch:",batch_count,"train loss:",loss.item(),"Original NRMSE:", NRMSE_loss(image_noise,ground_truth))
        
        loss.backward()

        with torch.no_grad():
            grad = sample_model.mask.grad
            grad = 2*torch.sigmoid((grad - torch.mean(grad))/torch.std(grad))-1
            grad = grad - torch.mean(grad)
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

    with torch.no_grad():
        loss = 0
        for val_batch in val_dataloader:
            val_batch.to(device)
            recon = recon_model(sample_model(val_batch).to(device))
            ground_truth = toIm(val_batch)
            loss += NRMSE_loss(recon.to(device),ground_truth.to(device))
        val_loss[epoch] = loss/len(val_dataloader.dataset)
        print("epoch:",epoch+1,"validation loss:",val_loss[epoch],"mask min:",torch.min(sample_model.mask),"mask max:",torch.max(sample_model.mask))

    torch.save(val_loss,"./unet_model_val_loss")
    torch.save(recon_model,"./unet_model")
    torch.save(sample_model.mask,"./unet_mask")
