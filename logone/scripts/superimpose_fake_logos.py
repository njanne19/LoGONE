import os 
import cv2 
import numpy as np 
from tqdm import tqdm 
from logone.scripts.place_fake_logo import place_logo
import yaml
from moviepy.editor import VideoFileClip, ImageSequenceClip
import pandas as pd


def main(video_dir, classlabel_path, fake_logopaths):

    # First load classlabels
    with open(classlabel_path, 'r') as f: 
        classlabels = yaml.safe_load(f)

    classlabels = classlabels['names']

    # Turn fake logopaths into fake logos, stored at video_dir/diffused/path 
    fake_logos = {}
    for class_name, fake_logopath in fake_logopaths.items():
        fake_logos[class_name] = cv2.imread(os.path.join(video_dir, 'diffused', fake_logopath))

    # Then we need to load the video, labeled original in the dir 
    video_path = os.path.join(video_dir, video_dir.split('/')[-1] + '_original' + '.MP4')

    # Load the video
    clip = VideoFileClip(video_path)

    # Get the best images for each class
    best_images = {k : cv2.imread(os.path.join(video_dir, 'best', f'class_{k}.jpg')) for k in fake_logopaths.keys()}
    last_homographies = {k : None for k in fake_logopaths.keys()}

    # Then we need to go frame by frame
    frames_with_description = tqdm(clip.iter_frames())
    edited_frames = []

    # For each frame, we need to extract the detections 
    for frame_index, frame in enumerate(frames_with_description): 
        frame_copy = frame.copy()
        frames_with_description.set_description(f"Frame {frame_index}")

        # Load the descriptions from video_dir/detections/frame_{i}.csv
        detection_path = os.path.join(video_dir, 'detections', f'frame_{frame_index}', f'detections.csv')
        detections = pd.read_csv(detection_path)

        # Iterate through the detections: 
        for detection_index, detection in detections.iterrows(): 
            class_id = detection['class']
            class_name = classlabels[class_id]

            # Get the bounding box 
            cx, cy, wx, wy = detection['x_center'], detection['y_center'], detection['width'], detection['height']
            w = frame.shape[1]
            h = frame.shape[0]
            # Convert to unnormalized coordinates 
            lx = int(cx*w - wx*w/2)
            ux = int(cx*w + wx*w/2)
            ly = int(cy*h - wy*h/2)
            uy = int(cy*h + wy*h/2)
            bbox= (lx, ly, ux, uy)

            # Get the logo in bounding box at video_dir/frames/frame_{frame_index}/detection_{detection_index}.jpg
            original_logo_path = os.path.join(video_dir, 'frames', f'frame_{frame_index}', f'detection_{detection_index}.jpg')
            original_logo = cv2.imread(original_logo_path)

            frame_copy, homography = place_logo(best_images[class_name], original_logo, frame_copy, fake_logos[class_name], bbox, last_homographies[class_name])
            
            # Save the new homography
            last_homographies[class_name] = homography

        edited_frames.append(frame_copy)

    # Finally, we need to save the edited video in video_dir/video_dir.split('/')[-1] + '_edited' + '.mp4'
    edited_video_path = os.path.join(video_dir, video_dir.split('/')[-1] + '_edited' + '.MP4')
    # Save with moviepy 
    edited_clip = ImageSequenceClip(edited_frames, fps=clip.fps)
    edited_clip = edited_clip.set_audio(clip.audio)
    edited_clip.write_videofile(edited_video_path, codec='libx264', audio_codec='aac')

    return 



if __name__ == "__main__": 

    video_dir = 'video_editing_data/MVI_1043'
    classlabel_path = 'logone/data/openlogo/yolo_finetune_split/data.yaml'

    fake_logopaths = {
        'fritos': 'fritos_c1.jpeg', 
        'mcdonalds': 'mcdonalds_c1.jpeg',
        'pepsi': 'pepsi_c1.jpeg',
        'reeses': 'reeses_c3.jpeg',
    }

    main(video_dir, classlabel_path, fake_logopaths)



