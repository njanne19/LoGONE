from torch.utils.data import Dataset
import numpy as np
import os
import cv2

class LogoTransformDataset(Dataset):
    def __init__(self,annotations_file, img_dir_t, img_dir_tt):
        self.img_paths = np.genfromtxt(annotations_file,
                                       delimiter=',', 
                                       dtype=str, 
                                       skip_header=1,
                                       usecols=(0,1))
        self.labels = np.genfromtxt(annotations_file,
                                       delimiter=',', 
                                       dtype=float, 
                                       skip_header=1,
                                       usecols=(0,1))
        self.img_dir_t = img_dir_t
        self.img_dir_tt = img_dir_tt

    def __len__(self):
        return len(os.listdir(self.img_dir_t))

    def __getitem__(self, idx):
        img_path_t = os.path.join(self.img_dir_t, self.img_paths[idx])
        img_path_tt = os.path.join(self.img_dir_tt, self.img_paths[idx])
        conc_img = np.concatenate((cv2.imread(img_path_t), cv2.imread(img_path_tt)), axis=2)
        return conc_img, self.labels[idx]

############# TESTING DATALOADER #############

from torch.utils.data import DataLoader

if __name__ == "__main__":
    training_data = LogoTransformDataset(os.path.join(os.getcwd(), 'logone', 'utilities', 'labels.csv'), 
                                         os.path.join(os.getcwd(), 'logone', 'utilities', 'transformed_256'), 
                                         os.path.join(os.getcwd(), 'logone', 'utilities', 'transformed_transformed_256'))
    test_data = LogoTransformDataset(os.path.join(os.getcwd(), 'logone', 'utilities', 'labels.csv'),
                                     os.path.join(os.getcwd(), 'logone', 'utilities', 'transformed_256'),
                                     os.path.join(os.getcwd(), 'logone', 'utilities', 'transformed_transformed_256'))
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
