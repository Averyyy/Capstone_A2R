from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch


class MyImageSet(Dataset):
    def __init__(self, filepath):
        super(MyImageSet).__init__()
        self.filepath = filepath
        self.img_list = [os.path.join(filepath, name) for name in 
        os.listdir(self.filepath) if '.jpg' in name]
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose((2, 0, 1))) #/ 127.5 - 1.
        # img = torch.unsqueeze(img, 0)
        return img
