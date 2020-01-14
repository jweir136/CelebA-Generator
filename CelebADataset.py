import torch
import torch.utils.data as data
from PIL import Image
import os
import sklearn.utils as utils

class CelebADataset(data.Dataset):
  def __init__(self, img_dir, df, transforms, training=True, shuffle=False):
    self.transforms = transforms
    self.img_dir = img_dir
    self.df = df

    if training:
      self.filenames = os.listdir(self.img_dir)[:150000]
    else:
      self.filenames = os.listdir(self.img_dir)[150000:]

    if shuffle:
      self.filenames = utils.shuffle(self.filenames)

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    img = Image.open(os.path.join(self.img_dir, self.filenames[idx]))
    X = self.transforms(img)
    Y = torch.tensor(self.df.loc[self.df['image_id'] == self.filenames[idx]].drop(['image_id'], axis=1).values)
    return X, Y
