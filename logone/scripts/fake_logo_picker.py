import os
import pandas as pd
from PIL import Image
import numpy as np
from place_fake_logo import place_logo

def load_images_and_csv(folder_path, original_images_folder):
    csv_file = os.path.join(folder_path, folder_path.split(os.sep)[-1] + '.csv')
    data = pd.read_csv(csv_file)

    images = {}
    previous_boxes = {}
    output_images = []
    
    for image_file in sorted(os.listdir(folder_path)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')) and image_file.startswith('help'):
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path)

            current_data = data[data['image_name'] == image_file]
            current_boxes = []

            for index, row in current_data.iterrows():
                bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
                cropped_image = image.crop(bbox)
                save_path = os.path.join(folder_path, f"cropped_{row['class_label']}_{index}.png")
                cropped_image.save(save_path)
                
                current_boxes.append((bbox, row['class_label'], save_path))

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