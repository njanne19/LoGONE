import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Directories for images
parent_dir = 'logone/data/openlogo/yolo_finetune_split'
train_images_dir = 'images/train'
val_images_dir = 'images/val'

# Directories for labels
train_labels_dir = 'labels/train'
val_labels_dir = 'labels/val'

# check to make sure val directories exist 
os.makedirs(os.path.join(parent_dir, val_images_dir), exist_ok=True)
os.makedirs(os.path.join(parent_dir, val_labels_dir), exist_ok=True)

# Number of files to copy
N = 100

# Get a list of image files
all_files = os.listdir(os.path.join(parent_dir, train_images_dir))

# Randomly select N files
selected_files = random.sample(all_files, N)

print(f"Copying files:")
selected_files_with_tqdm = tqdm(selected_files)
for file in selected_files_with_tqdm:
    selected_files_with_tqdm.set_description(f"Copying {file}")
    file = Path(file)
    img_to_copy = os.path.join(parent_dir, train_images_dir, file)
    label_to_copy = os.path.join(parent_dir, train_labels_dir, file.with_suffix('.txt'))
    img_dest = os.path.join(parent_dir, val_images_dir, file)
    label_dest = os.path.join(parent_dir, val_labels_dir, file.with_suffix('.txt'))
    shutil.copy(img_to_copy, img_dest)
    shutil.copy(label_to_copy, label_dest)

# # Copy selected image files
# for file in selected_files:
#     shutil.copy(os.path.join(train_images_dir, file), os.path.join(val_images_dir, file))

# # Copy corresponding label files
# for file in selected_files:
#     label_file = os.path.splitext(file)[0] + '.txt'
#     shutil.copy(os.path.join(train_labels_dir, label_file), os.path.join(val_labels_dir, label_file))
