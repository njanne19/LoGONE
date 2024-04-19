from torch.utils.data import Dataset
import numpy as np

class LogoTransformDataset(Dataset):
    def __init__(self,annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = np.loadtxt(annotations_file)