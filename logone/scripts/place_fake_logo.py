import cv2
import numpy as np
from find_image_transforms import align_images

def place_logo(image_one, image_two, image_for_scene, fake_logo, bbox_two):
    # Ensure the fake_logo has an alpha channel. If it doesn't, create one fully opaque.
    if fake_logo.shape[2] == 3:  # If no alpha channel
        fake_logo = cv2.cvtColor(fake_logo, cv2.COLOR_BGR2BGRA)
        fake_logo[:, :, 3] = 255  # Set alpha to fully opaque

    # Determine the size of the region to warp the logo into based on the bounding box of image_two
    h, w = bbox_two[3] - bbox_two[1], bbox_two[2] - bbox_two[0]

    # Create an output image for the warped logo with the same size as the ROI and fully transparent background
    warped_logo = np.zeros((h, w, 4), dtype=np.uint8)  # Initialize with transparent background

    homography = align_images(image_one, image_two)

    # Apply the known transformation matrix to warp the fake_logo
    warped_logo = cv2.warpPerspective(fake_logo, homography, (w, h), borderMode=cv2.BORDER_TRANSPARENT)

    # Extract the ROI from the scene image
    scene_roi = image_for_scene[bbox_two[1]:bbox_two[3], bbox_two[0]:bbox_two[2]]

    # Ensure the ROI is in BGRA for alpha blending
    if scene_roi.shape[2] == 3:
        scene_roi = cv2.cvtColor(scene_roi, cv2.COLOR_BGR2BGRA)

    # Blend the warped logo onto the scene ROI
    alpha_s = warped_logo[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(3):  # Iterate over the color channels
        scene_roi[:, :, c] = (alpha_s * warped_logo[:, :, c] + alpha_l * scene_roi[:, :, c]).astype('uint8')

    # Place the blended ROI back into the original scene image
    image_for_scene[bbox_two[1]:bbox_two[3], bbox_two[0]:bbox_two[2]] = scene_roi[:, :, :3]  # Assuming image_for_scene is without alpha

    return image_for_scene

# Example usage:
# Load your images using cv2.imread() here and define your bounding boxes
# image_one = cv2.imread('path_to_image_one.jpg')
# image_two = cv2.imread('path_to_image_two.jpg')
# image_for_scene = cv2.imread('path_to_image_for_scene.jpg')
# fake_logo = cv2.imread('path_to_fake_logo.jpg', cv2.IMREAD_UNCHANGED)  # Assuming it may or may not have alpha

# Define bounding boxes as tuples (x1, y1, x2, y2)
# bbox_one = (x1_one, y1_one, x2_one, y2_one)
# bbox_two = (x1_two, y1_two, x2_two, y2_two)

# Define the known transformation matrix
# transformation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)  # Example matrix

# Call the function
# result_image = place_logo(image_one, image_two, image_for_scene, fake_logo, bbox_one, bbox_two, transformation_matrix)
# cv2.imwrite('output.jpg', result_image)
