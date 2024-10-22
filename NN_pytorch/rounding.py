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

# %%
import scipy.io as sio

weight = torch.load("/home/wjy/Project/optsamp_model/opt100_mse_mask_snr3")
weight = weight.numpy()
sio.savemat('opt100_mse_mask_snr3.mat',{'weight':weight})
