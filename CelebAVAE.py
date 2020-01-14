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

  def encode(self, x):
    x = self.encoder(x)
    return x

  def forward(self, x):
    return x
