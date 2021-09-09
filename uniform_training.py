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

dataset = mri_data.SliceDataset(
    root=pathlib.Path(
      '/home/wjy/Project/fastmri_dataset/multicoil_test'
    ),
    transform=data_transform,
    challenge='multicoil'
)

# %% 