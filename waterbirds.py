import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class Waterbirds(Dataset):

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img = Image.open(self.df.img_filename.iloc[idx]).convert('RGB')
        bird_label = self.df.y.iloc[idx]
        water_label = self.df.place.iloc[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, bird_label, water_label