import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

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
trainloader = data.DataLoader(trainset, batch_size=32, num_workers=12, shuffle=True)
testloader = data.DataLoader(testset, batch_size=32, num_workers=12, shuffle=True)

##################### CREATE THE LOSS FUNCTION #####################################################

def loss_function(x, pred_x, mu, logvar):
  mse = fn.mse_loss(x, pred_x)
  kl = -5e-4 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
  return mse + kl 

###################### LOAD THE MODEL AND CREATE THE OPTIMIZER ##############################################################

vae = CelebAVAE()
vae = vae.cuda()

sgd = optim.Adam(vae.parameters(), lr=1e-3)

###################### TRAIN THE MODEL ##############################################################

for epoch in range(20):
  for x, y in tqdm(trainloader):
    sgd.zero_grad()

    x = x.cuda().float()

    x_pred, mu, logvar = vae.forward(x, training=True)

    train_loss = loss_function(x, x_pred, mu, logvar)

    train_loss.backward()
    sgd.step()

  for x, y in tqdm(testloader):
    with torch.no_grad():
      x = x.cuda().float()

      x_pred, mu, logvar = vae(x)

      test_loss = loss_function(x, x_pred, mu, logvar)

  print("\n")
  print("[{}] Train Loss={} Test Loss={}".format(epoch+1, train_loss.detach().cpu().numpy(), test_loss.detach().cpu().numpy()))
  print("\n")    

  # visualize the recontructions
  for x, y in testloader:
    x = x.cuda().float()
    x_pred, mu, logvar = vae(x)
    x_pred = x_pred.detach().cpu().numpy()[0]
    img = x.detach().cpu().numpy()[0]
 
    x_pred = np.moveaxis(x_pred, 0, -1)
    img = np.moveaxis(img, 0, -1)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(x_pred)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img)

    fig.savefig("reconstructions/epoch-{}.png".format(epoch+1))
 
    break

  fig.clf()

  # create a sample generation
  with torch.no_grad():
    sample = torch.randn(1, 2).cuda().float()
    generated_img = vae.decode(sample).detach().cpu().numpy()[0]
    generated_img = np.moveaxis(generated_img, 0, -1)
    plt.imshow(generated_img)
    plt.savefig("generations/epoch-{}.png".format(epoch+1))
