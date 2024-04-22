import os
import pandas as pd
from PIL import Image
import numpy as np
from place_fake_logo import place_logo

def load_images_and_annotations(folder_path, original_images_folder):
    images = {}
    previous_boxes = {}
    output_images = []
    
    for image_file in sorted(os.listdir(folder_path)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')) and image_file.startswith('MVI_1043'):
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path)
            annotation_file = image_file.rsplit('.', 1)[0] + '.txt'
            annotation_path = os.path.join(folder_path, annotation_file)

            if not os.path.exists(annotation_path):
                continue  # Skip if the annotation file does not exist
            
            current_boxes = []
            with open(annotation_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_label, xmin, ymin, xmax, ymax = parts
                        bbox = (int(xmin), int(ymin), int(xmax), int(ymax))
                        cropped_image = image.crop(bbox)
                        save_path = os.path.join(folder_path, f"cropped_{class_label}_{image_file}")
                        cropped_image.save(save_path)
                        
                        current_boxes.append((bbox, class_label, save_path))

            if previous_boxes:
                for current_box in current_boxes:
                    same_label_boxes = [box for box in previous_boxes if box[1] == current_box[1]]
                    if same_label_boxes:
                        closest_box = min(same_label_boxes, key=lambda x: np.linalg.norm(np.array(x[0][:2]) - np.array(current_box[0][:2])))
                        modified_image = place_logo(
                            image,
                            Image.open(closest_box[2]),  # First bounding boxed image from previous image
                            Image.open(current_box[2]),  # Second bounding boxed image from current image
                            closest_box[0] + current_box[0]  # Concatenate the bbox locations (tuple)
                        )
                        output_images.append(modified_image)
            else:
                original_image_path = os.path.join(original_images_folder, image_file)
                if os.path.exists(original_image_path):
                    original_image = Image.open(original_image_path)
                    # Additional handling for the first image as needed

            previous_boxes = current_boxes

    return output_images
