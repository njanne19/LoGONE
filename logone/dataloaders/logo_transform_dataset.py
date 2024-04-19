from torch.utils.data import Dataset
import numpy as np
import os

class LogoTransformDataset(Dataset):
    def __init__(self,annotations_file, img_dir_t, img_dir_tt):
        self.img_paths = np.genfromtxt(annotations_file,
                                       delimiter=',', 
                                       dtype=str, 
                                       skip_header=1,
                                       usecols=(0,1))
        self.img_paths = np.genfromtxt(annotations_file,
                                       delimiter=',', 
                                       dtype=float, 
                                       skip_header=1,
                                       usecols=(0,1))
        self.img_dir_t = img_dir_t
        self.img_dir_tt = img_dir_tt

    def __len__(self):
        return len(os.listdir(self.img_dir_t))

    def __getitem__(self, idx):
        img_path = 