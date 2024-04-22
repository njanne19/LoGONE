import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import normalize_logo_256


def load_and_save(original_img_filepath, transformed_img_filepath):
    """
    loads logo, transforms randomly, then saves transformed img accordingly, returns label to save
    """
    img = cv2.imread(original_img_filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        print('Couldnt load image: ', os.path.basename(original_img_filepath))
        return None, None, None

    img_256 = normalize_logo_256(img)
    cv2.imwrite(transformed_img_filepath, img_256)

def normalize_images_256(dataset_dir=os.path.join(os.getcwd(), 'test_images')):

    # get paths
    original_path=os.path.join(dataset_dir, 'original')
    transform_path=os.path.join(dataset_dir, 'original_256')

    # make directories
    if not os.path.exists(transform_path):
        os.mkdir(transform_path)

    # headers
    for imgfile in tqdm(os.listdir(original_path)):

        # name the img transformed with cylinder warping as name_t.ext
        name, ext = os.path.splitext(imgfile)
        imgfile_t = name + '_256' + ext 
        o_file = os.path.join(original_path, imgfile)
        t_file = os.path.join(transform_path, imgfile_t)
        load_and_save(o_file, t_file)

if __name__ == '__main__':
    normalize_images_256()