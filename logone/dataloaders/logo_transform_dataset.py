from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from logone.utilities.utils import zoom_to_bounding_box, normalize_logo_256, norm, normalize_logo_128, normalize_logo, build_corr
from logone.utilities.image_transformer import apply_logo_transform
# from utils import zoom_to_bounding_box

class LogoTransformDatasetCorr(Dataset):
    def __init__(self,annotations_file, img_dir, stat_path=None, normalize_img=True, normalize_labels=True):
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

        self.norm_label = normalize_labels
        self.norm_img = normalize_img
        self.img_dir = img_dir
        self.mean_img = 0
        self.std_img = 1
        self.mean_label = 0
        self.std_label = 1
        if stat_path is not None:
            norm_stats = np.loadtxt(stat_path)
            self.mean_img = norm_stats[0]
            self.std_img = norm_stats[1]
            self.mean_label = norm_stats[2]
            self.std_label = norm_stats[3]

        with open(annotations_file) as f:
            self.label_header = f.readline().strip('\n')

    def __len__(self):
        return self.img_paths.shape[0]

    def __getitem__(self, idx):
        img_path_t = os.path.join(self.img_dir, self.img_paths[idx,0])
        # imgt = normalize_logo_128(cv2.imread(img_path_t))
        imgt = normalize_logo(cv2.imread(img_path_t), 16)
        label = self.labels[idx].astype(np.float32)
        imgtt = apply_logo_transform(imgt, *label)
        if self.norm_img:
            imgt = norm(imgt, self.mean_img, self.std_img)
            imgtt = norm(imgtt, self.mean_img, self.std_img)
        if self.norm_label:
            label = norm(label, self.mean_label, self.std_label)

        # conc_img = np.concatenate((imgt, imgtt), axis=2).astype(np.float32)
        corr = build_corr(imgt, imgtt).astype(np.float32)
        return corr, label


class LogoTransformDataset(Dataset):
    def __init__(self,annotations_file, img_dir, stat_path=None, normalize_img=True, normalize_labels=True):
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

        self.norm_label = normalize_labels
        self.norm_img = normalize_img
        self.img_dir = img_dir
        self.mean_img = 0
        self.std_img = 1
        self.mean_label = 0
        self.std_label = 1
        if stat_path is not None:
            norm_stats = np.loadtxt(stat_path)
            self.mean_img = norm_stats[0]
            self.std_img = norm_stats[1]
            self.mean_label = norm_stats[2]
            self.std_label = norm_stats[3]

        with open(annotations_file) as f:
            self.label_header = f.readline().strip('\n')

    def __len__(self):
        return self.img_paths.shape[0]

    def __getitem__(self, idx):
        img_path_t = os.path.join(self.img_dir, self.img_paths[idx,0])
        img_path_tt = os.path.join(self.img_dir, self.img_paths[idx,1])
        imgt = normalize_logo_128(cv2.imread(img_path_t))
        # imgt = normalize_logo_256(cv2.imread(img_path_t))
        # imgtt = normalize_logo_256(cv2.imread(img_path_tt))
        label = self.labels[idx].astype(np.float32)
        imgtt = apply_logo_transform(imgt, *label)
        if self.norm_img:
            imgt = norm(imgt, self.mean_img, self.std_img)
            imgtt = norm(imgtt, self.mean_img, self.std_img)
        if self.norm_label:
            label = norm(label, self.mean_label, self.std_label)

        conc_img = np.concatenate((imgt, imgtt), axis=2).astype(np.float32)
        return conc_img, label


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