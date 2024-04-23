import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from find_image_transforms import align_images, resize_to_match
import pandas as pd 

def dist_from_center(x,y,h,w):
    diff = np.array([(w/2-y)/w,(h/2-x)/h])
    dist = np.sqrt(np.sum(diff**2)) / np.sqrt(2)
    if not (dist<1 and dist>0): print(x,y, h,w)
    return dist

def generate_alpha(img):
    scale = 2
    h, w = img.shape[:2]
    alpha = np.zeros_like(img[:,:,3], dtype=float)
    for x in range(h):
        for y in range(w):
            fade_alpha = 1- 1.0/float(min(min((h-x)/scale, x/scale), min((w-y)/scale, y/scale))+1)
            old_alpha = img[x,y,3] == 255
            alpha_new = min(fade_alpha, old_alpha)
            alpha[x,y] = alpha_new
            # alpha[x,y] = 1.0 - dist_from_center(x,y,h,w)
            # if alpha[x,y] < 0 or alpha[x,y] > 1: print(x,y,alpha[x,y])

    return alpha

def place_logo(image_one, image_two, image_for_scene, fake_logo, bbox_two, past_homography):
    """
    bbox_two = (lx, ly, ux, uy)
    """
    # Ensure the fake_logo has an alpha channel. If it doesn't, create one fully opaque.
    if fake_logo.shape[2] == 3:  # If no alpha channel
        fake_logo = cv2.cvtColor(fake_logo, cv2.COLOR_BGR2BGRA)
        fake_logo[:, :, 3] = 255  # Set alpha to fully opaque

    # Determine the size of the region to warp the logo into based on the bounding box of image_two
    h, w = abs(bbox_two[3] - bbox_two[1]), (bbox_two[2] - bbox_two[0])

    # Create an output image for the warped logo with the same size as the ROI and fully transparent background
    warped_logo = np.zeros((h, w, 4), dtype=np.uint8)  # Initialize with transparent background

    try: 
        homography = align_images(image_one, image_two)
        if homography is None:
            if past_homography is not None:
                homography = past_homography
            else:
                homography = np.eye(3, dtype=np.float32)
    except:
        if past_homography is not None:
            homography = past_homography
        else:
            homography = np.eye(3, dtype=np.float32)

    # Apply the known transformation matrix to warp the fake_logo
    fake_logo = cv2.resize(fake_logo, (w,h))
    warped_logo = cv2.warpPerspective(fake_logo, homography, (w,h), borderMode=cv2.BORDER_TRANSPARENT)[:,:,[2,1,0,3]]
    # fig, ax = plt.subplots(2)
    # ax[0].imshow(warped_logo)
    # ax[1].imshow(warped_logo[:,:,3])
    # plt.show()

    # Extract the ROI from the scene image
    scene_roi = image_for_scene[bbox_two[1]:bbox_two[3], bbox_two[0]:bbox_two[2]]

    # Ensure the ROI is in BGRA for alpha blending
    if scene_roi.shape[2] == 3:
        scene_roi = cv2.cvtColor(scene_roi, cv2.COLOR_BGR2BGRA)

    # Blend the warped logo onto the scene ROI
    # alpha_s = warped_logo[:, :, 3]
    alpha_s = generate_alpha(warped_logo)
    alpha_l = 1.0 - alpha_s
    for c in range(3):  # Iterate over the color channels
        scene_roi[:, :, c] = (alpha_s * warped_logo[:, :, c] + alpha_l * scene_roi[:, :, c]).astype('uint8')

    # Place the blended ROI back into the original scene image
    image_for_scene[bbox_two[1]:bbox_two[3], bbox_two[0]:bbox_two[2]] = scene_roi[:, :, :3]  # Assuming image_for_scene is without alpha

    return image_for_scene, homography

def extract_image_from_scene(img, labels):
    """
    """
    # f0,a0 = plt.subplots()
    # a0.imshow(img[:,:,[2,1,0]])
    h, w, c = img.shape
    n_instances = labels.shape[0]
    # _, axs = plt.subplots(n_instances)
    out = []
    for i in range(n_instances):
        id, cx, cy, wx, wy = labels[i]
        lx = int(cx*w - wx*w/2)
        ux = int(cx*w + wx*w/2)
        ly = int(cy*h - wy*h/2)
        uy = int(cy*h + wy*h/2)
        img_crop = img[ly:uy,lx:ux, :]
        bbox= (lx, ly, ux, uy)
        out.append((id, img_crop, bbox))
        # ax = axs[i]
        # ax.imshow(img_crop[:,:,[2,1,0]])
        # ax.set_title(str(id))

    # plt.tight_layout()
    # plt.show()
    return out

def stitch_images_to_video(image_list, output_video_path, fps=24):
    # Get dimensions of the first image
    height, width, _ = image_list[0].shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs based on your needs
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each image to the video
    for image in image_list:
        # image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        video_writer.write(image)

    # Release the video writer
    video_writer.release()


def fake_placer(data_dir = os.path.join(os.getcwd(), 'logone','test_data', 'train'), best_frame_idx=0):
    not_reeses_path = os.path.join(os.getcwd(), 'logone', 'test_data', 'fake_reeses.jpg')
    not_mcdonalds_path = os.path.join(os.getcwd(), 'logone', 'test_data', 'fake_mcdonalds.jpg')
    not_stella_path = os.path.join(os.getcwd(), 'logone', 'test_data', 'fake_stella.jpg')
    img_path = os.path.join(data_dir, 'images')
    label_path = os.path.join(data_dir, 'labels')
    frames = []
    labels = []
    print('loading images...')
    for i, file in tqdm(enumerate(os.listdir(img_path))):
        # if i > 300: break
        filename, ext = os.path.splitext(file)
        if i == best_frame_idx:
            first_frame = cv2.imread(os.path.join(img_path, filename + ext))
            first_label = np.loadtxt(os.path.join(label_path, filename + '.txt'))
        else:
            frames.append(cv2.imread(os.path.join(img_path, filename + ext)))
            labels.append(np.loadtxt(os.path.join(label_path, filename + '.txt')))

    # Example usage:
    # Load your images using cv2.imread() here and define your bounding boxes
    first_extraction = extract_image_from_scene(first_frame, first_label)

    fake_frames = []
    print('generating video...')
    for frame, label in tqdm(zip(frames, labels)):
        logos_two = extract_image_from_scene(frame, label)
        for first_id, first_logo, first_bbox in first_extraction:
            for next_id, next_logo, next_bbox in logos_two:
                res_img = frame
                if first_id == next_id:
                    if first_id == 5:
                        fake_logo = cv2.imread(not_reeses_path)[300:700,:,[2,1,0]]
                        res_img = place_logo(first_logo, next_logo, res_img, fake_logo, next_bbox)
                    elif first_id == 3:
                        fake_logo = cv2.imread(not_mcdonalds_path)[:,:,[2,1,0]]
                        res_img = place_logo(first_logo, next_logo, res_img, fake_logo, next_bbox)
                    elif first_id == 6:
                        fake_logo = cv2.imread(not_stella_path)[:,:,[2,1,0]]
                        res_img = place_logo(first_logo, next_logo, res_img, fake_logo, next_bbox)
        fake_frames.append(res_img)
        # fig, ax = plt.subplots()
        # ax.imshow(res_img[:,:,[2,1,0]])
        # plt.show()

    return fake_frames

if __name__ == "__main__": 
    fake_frames = fake_placer()
    stitch_images_to_video(fake_frames, 'fake_video.mp4')