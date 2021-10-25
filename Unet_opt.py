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


# %% sampling
factor = 8
sigma = 2
print("noise level:", sigma)
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

recon_model = torch.load('/project/jhaldar_118/jiayangw/OptSamp/warmup_model')

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
Loss = torch.nn.MSELoss()
# %% training
step = 1e-1
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
        support = torch.ge(ground_truth,0.06*ground_truth.max())
        recon = torch.mul(recon,support)
        ground_truth = torch.mul(ground_truth,support)

        loss = Loss(recon.to(device),ground_truth.to(device))
        if batch_count%100 == 0:
            print("batch:",batch_count,"train MSE:",loss.item(),"Original MSE:", Loss(recon,ground_truth))
 
   
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

    with torch.no_grad():
        loss = 0
        orig_loss = 0
        for val_batch in val_dataloader:
            val_batch.to(device)

            image_noise = sample_model(val_batch)
            recon = recon_model(image_noise.to(device))
            ground_truth = toIm(val_batch)
            support = torch.ge(ground_truth,0.06*ground_truth.max())
            recon = torch.mul(recon,support)
            ground_truth = torch.mul(ground_truth,support)

            loss += Loss(recon.to(device),ground_truth.to(device))
            orig_loss += Loss(recon.to(device),ground_truth.to(device))

        val_loss[epoch] = loss/len(val_dataloader)
        print("epoch:",epoch+1,"validation MSE:",val_loss[epoch],"original MSE:",orig_loss/len(val_dataloader))

    torch.save(val_loss,"./unet_model_val_loss_noise"+str(sigma))
    torch.save(recon_model,"./unet_model_noise"+str(sigma))
    torch.save(sample_model.mask,"./unet_mask_noise"+str(sigma))
