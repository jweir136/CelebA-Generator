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

df = pd.read_csv("../../datasets/celebA/list_attr_celeba.csv")
img_dir = "../../datasets/celebA/img_align_celeba/img_align_celeba"

trans = transforms.Compose([
  transforms.Resize(100),
  transforms.ToTensor()
])

trainset = CelebADataset(img_dir, df, trans, training=True)
testset = CelebADataset(img_dir, df, trans, training=False)
trainloader = data.DataLoader(trainset, batch_size=128, num_workers=12)
testloader = data.DataLoader(testset, batch_size=128, num_workers=12)

mean = 0.0
for images, _ in tqdm(trainloader):
    batch_samples = images.size(0) 
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)

mean = mean / len(trainloader.dataset)

var = 0.0
for images, _ in tqdm(trainloader):
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    var += ((images - mean.unsqueeze(1))**2).sum([0,2])

std = torch.sqrt(var / (len(trainloader.dataset)*100*100))

np.save("mean.npy", mean)
np.save("std.npy", std)
