import yaml
import pandas as pd
import os
import cv2
from tqdm import tqdm

old_yaml_filename = 'logone/data/openlogo/yolo_finetune_split/data_old.yaml'
new_yaml_filename = 'logone/data/openlogo/yolo_finetune_split/data.yaml'

with open(old_yaml_filename, 'r') as file:
    old_yaml_file = yaml.safe_load(file)

with open(new_yaml_filename, 'r') as file:
    new_yaml_file = yaml.safe_load(file)

new_classdict_num_to_class = new_yaml_file['names']
old_classdict_num_to_class = {i: class_name for i, class_name in enumerate(old_yaml_file['names'])}

def invert_dict(d):
    return {v: k for k, v in d.items()}

new_classdict_class_to_num = invert_dict(new_classdict_num_to_class)
old_classdict_class_to_num = invert_dict(old_classdict_num_to_class)

def preview_convert_single_text_file(text_filename): 

    # Iterates through the rows of a text file. 
    # Grabs the first number in each row of the file, and passes through 
    # old_classdict_num_to_class to get the class name.
    # Then, passes through new_classdict_class_to_num to get the new class number.
    # Then prints the output here 

    with open(text_filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_num = int(line.split()[0])
            class_name = old_classdict_num_to_class[class_num]
            new_class_num = new_classdict_class_to_num[class_name]
            print(f'Old class number: {class_num} | Old class name: {class_name} | New class number: {new_class_num}')


def old_class_num_to_new_class_num(old_class_num):
    old_class_name = old_classdict_num_to_class[old_class_num]
    new_class_num = new_classdict_class_to_num[old_class_name]
    return new_class_num

def render_single_frame_with_old_annotations(filename): 
    # filename exists as filename.txt in labels/train and filename.jpg in images/train
    # filename.txt contains annotations in the format:
    # class_number x_center y_center width height (normalized) 
    # where num rows = num objects in image
    # filename.jpg is the image file

    # Read the text file
    # Create temp dataframe with the annotation
    text_file = os.path.join('logone/data/openlogo/yolo_finetune_split/labels/train', filename + '.txt')
    image_file = os.path.join('logone/data/openlogo/yolo_finetune_split/images/train', filename + '.jpg')

    df = pd.read_csv(text_file, sep=' ', header=None)

    # Remap first column of dataframe to new class numbers
    df[0] = df[0].apply(old_class_num_to_new_class_num)

    # Read the image file
    image = cv2.imread(image_file)

    # Draw the detection boxes 
    for i, row in df.iterrows(): 
        class_num, x_center, y_center, width, height = row
        x_center = int(x_center * image.shape[1])
        y_center = int(y_center * image.shape[0])
        width = int(width * image.shape[1])
        height = int(height * image.shape[0])

        x1 = int(x_center - width/2)
        y1 = int(y_center - height/2)
        x2 = int(x_center + width/2)
        y2 = int(y_center + height/2)

        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add text
        image = cv2.putText(image, new_classdict_num_to_class[class_num], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    imS = cv2.resize(image, (960, 540))
    cv2.imshow('image', imS)
    cv2.waitKey(0)

def render_single_frame_with_new_annotations(filename): 
    # filename exists as filename.txt in labels/train and filename.jpg in images/train
    # filename.txt contains annotations in the format:
    # class_number x_center y_center width height (normalized) 
    # where num rows = num objects in image
    # filename.jpg is the image file

    # Read the text file
    # Create temp dataframe with the annotation
    text_file = os.path.join('logone/data/openlogo/yolo_finetune_split/labels/train', filename + '.txt')
    image_file = os.path.join('logone/data/openlogo/yolo_finetune_split/images/train', filename + '.jpg')

    df = pd.read_csv(text_file, sep=' ', header=None)

    # Read the image file
    image = cv2.imread(image_file)

    # Draw the detection boxes 
    for i, row in df.iterrows(): 
        class_num, x_center, y_center, width, height = row
        x_center = int(x_center * image.shape[1])
        y_center = int(y_center * image.shape[0])
        width = int(width * image.shape[1])
        height = int(height * image.shape[0])

        x1 = int(x_center - width/2)
        y1 = int(y_center - height/2)
        x2 = int(x_center + width/2)
        y2 = int(y_center + height/2)

        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add text
        image = cv2.putText(image, new_classdict_num_to_class[class_num], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    imS = cv2.resize(image, (960, 540))
    cv2.imshow('image', imS)
    cv2.waitKey(0)
    

# preview_convert_single_text_file('/home/nick/Documents/LoGONE/logone/data/openlogo/yolo_finetune_split/labels/train/drunkDrive_mp4-0000_jpg.rf.824b7341345f3941e62ac7220698133c.txt')
# render_single_frame_with_annotations('MVI_1043_MP4-0243_jpg.rf.aa7e5e7d5beca64d50749f551d5483b4')

def convert_text_file_to_new_classes(text_filename): 
    # Iterates through one text file and converts to new class indexing 
    # Writes the new text file to the same directory with the same name

    # Parse with pandas dataframe
    df = pd.read_csv(text_filename, sep=' ', header=None)

    # Remap first column of dataframe to new class numbers
    df[0] = df[0].apply(old_class_num_to_new_class_num)

    # Write to the orginal file
    df.to_csv(text_filename, sep=' ', header=False, index=False)


def convert_all_text_files_to_new_classes(directory): 
    # Iterates through all text files in a directory and converts to new class indexing
    directory_with_tqdm = tqdm(os.listdir(directory))
    for filename in directory_with_tqdm: 
        directory_with_tqdm.set_description(f'Converting {filename}')
        if filename.endswith('.txt'): 
            convert_text_file_to_new_classes(os.path.join(directory, filename))


# convert_all_text_files_to_new_classes('logone/data/openlogo/yolo_finetune_split/labels/train')
render_single_frame_with_new_annotations('moveAround_mp4-0270_jpg.rf.20624fc9c4b338526d360f293be26406')