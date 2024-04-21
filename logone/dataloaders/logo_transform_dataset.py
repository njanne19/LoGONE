from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from logone.utilities.utils import zoom_to_bounding_box
# from utils import zoom_to_bounding_box

class LogoTransformDataset(Dataset):
    def __init__(self,annotations_file, img_dir):
        self.img_paths = np.genfromtxt(annotations_file,
                                       delimiter=',', 
                                       dtype=str, 
                                       skip_header=1,
                                       usecols=(0,1))
        self.labels = np.genfromtxt(annotations_file,
                                    delimiter=',',
                                    dtype=float,
                                    skip_header=1,
                                    usecols=(2,3,4,5,6,7,8,9,10))
        self.img_dir = img_dir
        with open(annotations_file) as f:
            self.label_header = f.readline().strip('\n')

    def __len__(self):
        return self.img_paths.shape[0]

    def __getitem__(self, idx):
        img_path_t = os.path.join(self.img_dir, self.img_paths[idx,0])
        img_path_tt = os.path.join(self.img_dir, self.img_paths[idx,1])
        conc_img = np.concatenate((zoom_to_bounding_box(cv2.imread(img_path_t)), zoom_to_bounding_box(cv2.imread(img_path_tt))), axis=2)
        return conc_img.astype(np.float32), self.labels[idx]


# ############# TESTING DATALOADER #############

from torch.utils.data import DataLoader

if __name__ == "__main__":
    train_data = LogoTransformDataset(os.path.join(os.getcwd(), 'logone', 'utilities', 'labels_train.csv'), 
                                         os.path.join(os.getcwd(), 'logone', 'utilities', 'transformed_256'))
    test_data = LogoTransformDataset(os.path.join(os.getcwd(), 'logone', 'utilities', 'labels_test.csv'),
                                     os.path.join(os.getcwd(), 'logone', 'utilities', 'transformed_256'))
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    for test_images, test_labels in test_dataloader:
        print(test_images[0].shape, test_labels[0])