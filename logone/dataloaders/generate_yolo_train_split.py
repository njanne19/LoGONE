import fiftyone as fo 
import yaml
import os 
import numpy as np 
import torch
import argparse
import shutil
from tqdm import tqdm

# TODO: Add support for single class mode

# Use these seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def main(args): 

    ################ Arguments ################

    # First try and get the dataset 
    try: 
        dataset = fo.load_dataset(args.dataset_name)
    except: 
        print(f"Could not find dataset with name {args.dataset_name}. Exiting.")
        return
    
    # Get the parent directory of the FiftyOne dataset 
    print(f"Finding parent directory of dataset {args.dataset_name}...")
    random_sample = dataset.take(1).first()
    sample_filepath = random_sample.filepath 

    # The filepath has a separation .../data/DATASET_NAME/..., we want to find dataset name 
    try: 
        split_filepath = sample_filepath.split('/')
        dataset_name = split_filepath[split_filepath.index('data')+1]
        current_file_directory = os.path.abspath(__file__)
        dataset_parent_directory = os.path.abspath(os.path.join(current_file_directory, os.pardir, os.pardir, 'data', dataset_name))
        print(f"Found dataset parent directory at {dataset_parent_directory}")
    except Exception as e: 
        print(f"Could not find parent directory of dataset {args.dataset_name}. Exiting.")
        print(f"[NOTE]: Sample filepath returned {sample_filepath}")
        print(e)
        return
    
    # Get the split name 
    split_name = args.split_name

    # Check to see if this split already exists, if it does then append an integer to the end until it agree
    if os.path.exists(os.path.join(dataset_parent_directory, split_name)): 
        print(f"Split {split_name} already exists. Appending integer to split name...")
        i = 1
        while os.path.exists(os.path.join(dataset_parent_directory, f"{split_name}{i}")): 
            i += 1
        split_name = f"{split_name}{i}"
        print(f"New split name is {split_name}")
    
    print(f"Creating split {split_name} in {dataset_parent_directory}...")
    split_directory = os.path.join(dataset_parent_directory, split_name)

    # Create the split directory
    try: 
        os.makedirs(split_directory)
    except Exception as e: 
        print(f"Could not create split directory {split_directory}. Exiting.")
        print(e)
        return

    # Get the validation split size and single class mode
    val_split = args.val_split
    single_class = args.single_class

    ################## Data Collection ##################

    # First thing we're going to do is collect information we need for the YOLO training from FiftyOne 
    print("Collecting information from FiftyOne dataset...")
    num_samples = dataset.count()
    image_filenames = []
    in_test_set = np.empty(num_samples, dtype=bool)
    annotations = {}
    unique_classes = set()

    dataset_with_progress = tqdm(dataset)
    for sample_index, sample in enumerate(dataset_with_progress):
        # dataset_with_progress.set_description(f"Processing {sample.filepath[-30:]:>30.30}...")
        image_filenames.append(sample.filepath)
        dataset_with_progress.set_description(f"Processing {image_filenames[sample_index][-30:]:>30.30}...")

        if sample["ground_truth"] is not None: 
            annotations[sample.filepath] = sample["ground_truth"].detections
            # Get the unique classes 
            for detection in sample["ground_truth"].detections: 
                unique_classes.add(detection.label)

        # print(f"Sample tags: {sample.tags}")

        if 'default_split/test' in sample.tags: 
            in_test_set[sample_index] = True
        else: 
            in_test_set[sample_index] = False
        
    # First do some processing on the unique classes 
    unique_classes = list(unique_classes)
    num_classes = len(unique_classes)
    # then sort 
    unique_classes.sort()
    # Then form a dict, class_dit is {"class_name": class_index}, invert_class_dict is {class_index: "class_name"} 
    class_dict = {unique_classes[i]: i for i in range(num_classes)}
    invert_class_dict = {i: unique_classes[i] for i in range(num_classes)}

    # Then generate train/val/test splits 
    all_train_indices = np.where(in_test_set == False)[0]
    all_train_indices = np.random.permutation(all_train_indices) # Shuffle for split generation

    # Get the number of samples in the training set
    num_train_samples = len(all_train_indices)
    num_val_samples = int(val_split * num_train_samples)

    print(f"Total number of training samples: {num_train_samples}")
    print(f"Number of samples taken for validation: {num_val_samples}")

    # Split the training set into train and validation sets
    train_split, val_split = all_train_indices[:-num_val_samples], all_train_indices[-num_val_samples:]

    # Then collect the actual filenames from the indices
    image_filenames = np.array(image_filenames)
    train_split, val_split, test_split = image_filenames[train_split], image_filenames[val_split], image_filenames[in_test_set]

    ################## File Writing ##################
    os.makedirs(os.path.join(split_directory, 'images'))
    os.makedirs(os.path.join(split_directory, 'images', 'train'))
    os.makedirs(os.path.join(split_directory, 'images', 'test'))
    os.makedirs(os.path.join(split_directory, 'labels'))
    os.makedirs(os.path.join(split_directory, 'labels', 'train'))
    os.makedirs(os.path.join(split_directory, 'labels', 'test'))

    if len(val_split) != 0: 
        os.makedirs(os.path.join(split_directory, 'images', 'val'))
        os.makedirs(os.path.join(split_directory, 'labels', 'val'))

    # Now we need to write copies of the image files to the split directory 
    print("Copying image files to split directory...")
    for current_split, current_split_name in zip([train_split, val_split, test_split], ['train', 'val', 'test']):
        split_with_progress = tqdm(current_split)
        for image_file in split_with_progress: 
            split_with_progress.set_description(f"Copying {image_file:>30.30}...")
            
            # Copy the image file to the directory 
            image_file_name = os.path.basename(image_file)

            # Copy the image file
            shutil.copy(image_file, os.path.join(split_directory, 'images', current_split_name, image_file_name))

            # Then create the associated text file 
            # To do so, we iterate over the annotations of this file, we write a line 
            # to a .txt file with class_index, x_center, y_center, width, height (normalized coordinates, FiftyOne format)

            # Get the annotations for this image
            if image_file in annotations:
                image_annotations = annotations[image_file]
            else: 
                continue # Don't write a text file if there are no annotations

            # Create the text file
            text_file_name = image_file_name.replace('.jpg', '.txt')

            with open(os.path.join(split_directory, 'labels', current_split_name, text_file_name), 'w') as f:
                for annotation in image_annotations: 
                    # Get the class index 
                    class_index = class_dict[annotation.label]

                    # Get the bounding box 
                    x_min = annotation.bounding_box[0] 
                    y_min = annotation.bounding_box[1]
                    width = annotation.bounding_box[2]
                    height = annotation.bounding_box[3]

                    x_center = x_min + width / 2
                    y_center = y_min + height / 2

                    # Write to the file 
                    f.write(f"{class_index} {x_center} {y_center} {width} {height}\n")


    # We need to write a .yaml file to record class indices, split paths, etc. 
    # Start by opening a new yaml file inside the split directory
    with open(os.path.join(split_directory, 'data.yaml'), 'w') as f:
        data = {
            "path": split_directory,
            "train": 'images/train', 
            "val": 'images/val',
            "test": 'images/test',
            "names": invert_class_dict
        }
        yaml.dump(data, f, default_flow_style=False)

    # Finally, need to add the split to the FiftyOne dataset 
    print("Adding split to FiftyOne dataset...")
    # Iterate over the dataset, add the split tag to the appropriate samples
    dataset_with_progress = tqdm(dataset)
    for sample in dataset_with_progress:
        dataset_with_progress.set_description(f"Processing {sample.filepath:>30.30}...")
        if sample.filepath in train_split: 
            sample.tags.append((split_name + '/train'))
        elif sample.filepath in val_split: 
            sample.tags.append((split_name + '/val'))
        elif sample.filepath in test_split: 
            sample.tags.append((split_name + '/test'))

        sample.save()



if __name__ == "__main__": 

    # Create an argument parser 
    parser = argparse.ArgumentParser(description=
    """
    Generates train/val/test spits for YOLO training using ultralytics. At this point, a FiftyOne dataset 
    should be loaded locally (by calling load_*_as_fiftyone.py in dataloaders/). 

    The caller should specify the name of the dataset (FiftyOne). 

    Optional parameters allow the caller to add a parameterized validation split, name the split, and 
    specify whether or not to use "single class mode" (i.e. all logos are of the same class) or multi-class mode, 
    where logo labels are inferred from the FiftyOne dataset. This script results in modifications to the FiftyOne 
    annotations, as well as the CREATION and COPY of data splits in the respective folders.
                                     
    """)

    # 'dataset_name' argument, required 
    parser.add_argument('dataset_name', type=str, help='Name of the FiftyOne dataset to use for YOLO training.')

    # 'val_split' argument, float, optional 
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split size. Default is 0.1. In this example, \
                        10 percent of the training data will be used for validation.')
    
    # 'single_class' argument, boolean, optional 
    parser.add_argument('--single_class', type=bool, default=False, help='If True, all logos are of the same class. If False, \
                        logos are inferred from the FiftyOne dataset. Default is False.')

    # 'split_name' argument, string, optional
    parser.add_argument('--split_name', type=str, default='yolo_split', help='Name of the split. Default is yolo_split[n].')

    # Parse the arguments 
    args = parser.parse_args() 

    main(args) 