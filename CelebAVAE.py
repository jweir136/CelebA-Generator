import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

class Flatten(nn.Module):
  def forward(self, x):
    batch_size = x.size()[0]
    return x.view(batch_size, 128*84*84)

class UnFlatten(nn.Module):
  def forward(self, x):
    batch_size = x.size()[0]
    return x.view(batch_size, 128, 84, 84)

class CelebAVAE(nn.Module):
  def __init__(self):
    super().__init__()

    self.drop = nn.Dropout2d(p=0.2, inplace=True)

    self.encoder = nn.Sequential(
      nn.Conv2d(3, 16, 5),
      nn.ReLU(True),
      nn.Conv2d(16, 32, 5),
      nn.ReLU(True),
      nn.Conv2d(32, 64, 5),
      nn.ReLU(True),
      nn.Conv2d(64, 128, 5),
      nn.ReLU(True),
      Flatten(),
      nn.Linear(128*84*84, 120),
      nn.ReLU(True)
    )
    self.mu_layer = nn.Linear(120, 2)
    self.logvar_layer = nn.Linear(120, 2)
    self.decoder = nn.Sequential(
      nn.Linear(2, 120),
      nn.ReLU(True),
      nn.Linear(120, 128*84*84),
      nn.ReLU(True),
      UnFlatten(),
      nn.ConvTranspose2d(128, 64, 5),
      nn.ReLU(True),
      nn.ConvTranspose2d(64, 32, 5),
      nn.ReLU(True),
      nn.ConvTranspose2d(32, 16, 5),
      nn.ReLU(True),
      nn.ConvTranspose2d(16, 3, 5),
      nn.Tanh()
    )

  def __reparam__(self, mu, logvar):
    std = 0.5 * torch.exp(logvar)
    epsilon = torch.rand_like(std)
    return mu + std * epsilon

  def encode(self, x):
    x = self.encoder(x)
    mu, logvar = self.mu_layer(x), self.logvar_layer(x)
    return x, mu, logvar

  def decode(self, x):
    return self.decoder(x)

  def forward(self, x, training=False):
    if training:
      x = self.drop(x)
    x, mu, logvar = self.encode(x)
    z = self.__reparam__(mu, logvar)
    x = self.decoder(z)
    return x, mu, logvar
