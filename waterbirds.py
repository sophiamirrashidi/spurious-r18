import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class Waterbirds(Dataset):

    def __init__(self, df, root, transform=None):
        self.df = df
        self.root = root
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.df.img_filename.iloc[idx])
        img = Image.open(img_path).convert('RGB')
        bird_label = self.df.y.iloc[idx]
        water_label = self.df.place.iloc[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, bird_label, water_label