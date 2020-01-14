import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from CelebADataset import *
from CelebAVAE import *

###################### CREATE THE TRANSFORMATIONS, DATASETS, AND DATA LOADERS #####################

df = pd.read_csv("../../datasets/celebA/list_attr_celeba.csv")
img_dir = "../../datasets/celebA/img_align_celeba/img_align_celeba"

mean = np.load("mean.npy")
std = np.load("std.npy")

trans = transforms.Compose([
  transforms.Resize((100, 100)),
  transforms.ToTensor(),
  transforms.Normalize(mean=mean, std=std)
])

trainset = CelebADataset(img_dir, df, trans, training=True, shuffle=True)
testset = CelebADataset(img_dir, df, trans, training=False, shuffle=True)
trainloader = data.DataLoader(trainset, batch_size=1, num_workers=12)
testloader = data.DataLoader(testset, batch_size=128, num_workers=12)

###################### LOAD THE MODEL ##############################################################

vae = CelebAVAE()
vae = vae.cuda()

for x, _ in trainloader:
  x = x.cuda().float()
  print(vae.encode(x).size())
  break
