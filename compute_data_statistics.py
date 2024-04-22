import os
import numpy as np
from torch.utils.data import DataLoader
from logone.dataloaders.logo_transform_dataset import LogoTransformDataset

def check_normalization():
    data_dir = os.path.join(os.getcwd(), 'logone', 'utilities')
    stat_path = os.path.join(data_dir, 'data_norm_stats.csv')
    val_data = LogoTransformDataset(os.path.join(data_dir, 'labels_corrected.csv'),
                                    os.path.join(data_dir, 'transformed_256'),
                                    stat_path=stat_path)
    n_data = len(val_data)
    val_dataloader = DataLoader(val_data, batch_size=n_data, shuffle=True)
    data = next(iter(val_dataloader))

    mean_imgs = data[0].mean().numpy()
    std_imgs = data[0].std().numpy()
    mean_labels = data[1].mean().numpy()
    std_labels = data[1].std().numpy()

    norm_stats = np.array([mean_imgs, std_imgs, mean_labels, std_labels])
    print('norm stats: ', norm_stats)

def label_stats():
    data_dir = os.path.join(os.getcwd(), 'logone', 'utilities')
    val_data = LogoTransformDataset(os.path.join(data_dir, 'labels_corrected.csv'),
                                    os.path.join(data_dir, 'transformed_256'))
    n_data = len(val_data)
    val_dataloader = DataLoader(val_data, batch_size=n_data, shuffle=True)
    data = next(iter(val_dataloader))

    mean_imgs = data[0].mean().numpy()
    std_imgs = data[0].std().numpy()
    mean_labels = data[1].mean().numpy()
    std_labels = data[1].std().numpy()

    norm_stats = np.array([mean_imgs, std_imgs, mean_labels, std_labels])
    print('Saving Norm Stats: ', norm_stats)

    np.savetxt(os.path.join(data_dir, 'data_norm_stats.csv'), norm_stats)


if __name__ == "__main__":
    # label_stats()
    check_normalization()