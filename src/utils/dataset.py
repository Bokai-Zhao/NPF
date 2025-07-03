import torch
from torch.utils.data import Dataset
import numpy as np

def get_he_tile(img,x,y,size,padding_value=(0,0,0)):
    x = int(x)
    y = int(y)
    left = x - size//2
    right = left + size
    upper = y - size//2
    lower = upper + size

    left_crop = max(0, left)
    upper_crop = max(0, upper)
    right_crop = min(img.shape[1], right)
    lower_crop = min(img.shape[0], lower)

    pad_top = max(0, abs(upper_crop - upper))
    pad_bottom = max(0, abs(lower - lower_crop))
    pad_left = max(0, abs(left_crop - left))
    pad_right = max(0, abs(right - right_crop))

    tile = img[upper_crop:lower_crop, left_crop:right_crop,:]
    new_height = tile.shape[0] + pad_top + pad_bottom
    new_width = tile.shape[1] + pad_left + pad_right
    padded_tile = np.full((new_height, new_width, tile.shape[2]), padding_value, dtype=tile.dtype)
    padded_tile[pad_top:pad_top+tile.shape[0], pad_left:pad_left+tile.shape[1]] = tile
    return padded_tile

class SampleDataset(Dataset):
    def __init__(self, adata,img,size, transform=None):
        self.adata = adata
        self.img = img
        self.transform = transform
        self.size = size
        self.x_max = img.shape[1]
        self.y_max = img.shape[0]
    
    def __len__(self):
        return len(self.adata)
    
    def __getitem__(self, idx):
        protein_data = torch.tensor(self.adata.X[idx],dtype=torch.float32)
        x,y = self.adata.obsm['spatial'][idx]
        imgdata = get_he_tile(self.img,x,y,self.size)
        if self.transform:
            imgdata = self.transform(imgdata)
        x = (x-self.x_max//2)/self.x_max*2
        y = (y-self.y_max//2)/self.y_max*2
        posion_data = torch.tensor([x,y],dtype=torch.float32)
        return imgdata,posion_data,protein_data

