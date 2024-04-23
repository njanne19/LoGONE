from ultralytics import YOLO
import os 
import pandas as pd
import shutil
from moviepy.editor import VideoFileClip, ImageSequenceClip
from tqdm import tqdm 
from pathlib import Path 
import yaml
import cv2
from PIL import Image


def main(args): 

    # Define the YOLO model 
    model = YOLO(args['weights_file'])

    # Create the output directory if it doesn't already exist 
    os.makedirs(args['output_folder'], exist_ok=True)

    # Get name of video file without extension 
    video_filename_base = Path(args['video_file']).stem
    # Make a subdirectory in the output folder with the video filename
    video_output_folder = os.path.join(args['output_folder'], video_filename_base)
    os.makedirs(video_output_folder, exist_ok=True)

    # Load the video file 
    video = VideoFileClip(args['video_file'])

    # Create a directory in the output folder to store the frames and detections 
    frames_folder = os.path.join(video_output_folder, 'frames')
    os.makedirs(frames_folder, exist_ok=True)

    detections_folder = os.path.join(video_output_folder, 'detections')
    os.makedirs(detections_folder, exist_ok=True)

    best_folder = os.path.join(video_output_folder, 'best')
    os.makedirs(best_folder, exist_ok=True)

    detections_on_frame_index = {}
    processed_frames = []
    unique_classes = set()

    # Iterate through the frames, get best detections, and save the frames
    print(f"Running inference on video {args['video_file']}...")
    frames_with_tqdm = tqdm(video.iter_frames())
    for i, frame in enumerate(frames_with_tqdm): 
        frames_with_tqdm.set_description(f"Frame {i}")

        # Get the BGR image from the RGB 
        bgr_frame = frame[:, :, ::-1]

        # Get the detections 
        results = model(bgr_frame)[0] 

        # Process the annotated frame 
        annotated_frame_bgr = results.plot() 
        annotated_frame_rgb = annotated_frame_bgr[:, :, ::-1]
        processed_frames.append(annotated_frame_rgb)

        # Get bounding boxes
        bboxes = results.boxes
        class_labels = bboxes.cls 
        normalized_xywh = bboxes.xywhn
        conf = bboxes.conf # Confidence scores 

        # Convert all to numpy 
        class_labels = class_labels.cpu().numpy()
        normalized_xywh = normalized_xywh.cpu().numpy()
        conf = conf.cpu().numpy()

        # Create a new pandas dataframe to store the detections 
        df = pd.DataFrame({
            'class': class_labels.astype(int),
            'x_center': normalized_xywh[:, 0],
            'y_center': normalized_xywh[:, 1],
            'width': normalized_xywh[:, 2],
            'height': normalized_xywh[:, 3],
            'confidence': conf
        })

        # Make frame specific folder for detections
        frame_folder = os.path.join(detections_folder, f"frame_{i}")
        os.makedirs(frame_folder, exist_ok=True)

        # Save the detections to a csv file
        df.to_csv(os.path.join(frame_folder, 'detections.csv'), index=False)

        # Make frame specific folder for frames 
        frame_specific_folder = os.path.join(frames_folder, f"frame_{i}")
        os.makedirs(frame_specific_folder, exist_ok=True)

        detections_on_frame_index[i] = {}
        detections_on_frame_index[i]["annotations"] = df
        detections_on_frame_index[i]["frames"] = []

        # Then get all the detections as cropped images of the original 
        for j, detection in enumerate(class_labels): 
            unique_classes.add(int(detection))
            x_center, y_center, width, height = normalized_xywh[j]
            x_center = int(x_center * frame.shape[1])
            y_center = int(y_center * frame.shape[0])
            width = int(width * frame.shape[1])
            height = int(height * frame.shape[0])

            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)

            cropped_image = frame[y1:y2, x1:x2]
            cropped_image_pil = Image.fromarray(cropped_image)
            cropped_image_path = os.path.join(frame_specific_folder, f"detection_{j}.jpg")
            cropped_image_pil.save(cropped_image_path)
            detections_on_frame_index[i]["frames"].append(cropped_image_path)

    # Now we need to find the best detection for each class across all frames 
    # Start by parsing the yaml file to get index -> class mapping
    with open(args['class_reference_file'], 'r') as file:
        class_reference = yaml.safe_load(file)

    # Get the class labels
    class_labels_mapping = class_reference['names']

    print(f"Finding best detections for each class...")
    for class_label in unique_classes:
        best_detection_csv = None
        best_detection = None
        best_confidence = 0

        for frame_index, frame_data in detections_on_frame_index.items():
            print(f"For class {class_labels_mapping[class_label]} in frame {frame_index}...")
            candidate_annotations = frame_data["annotations"]
            candidate_frames = frame_data["frames"]

            # Get the rows in the annotations dataframe that corresponds to the class label
            mask = candidate_annotations['class'] == class_label
            class_annotations = candidate_annotations[mask]

            # If there are no annotations for this class in this frame, skip
            if class_annotations.empty:
                continue

            # Iterate through the annotations for this class in this frame
            for i, row in class_annotations.iterrows():
                confidence = row['confidence']
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_detection = candidate_frames[i]
                    best_detection_csv = os.path.join(detections_folder, f"frame_{frame_index}", 'detections.csv')

        # Save the best detection to the best folder
        best_detection_pil = Image.open(best_detection)
        best_detection_pil.save(os.path.join(best_folder, f"class_{class_labels_mapping[class_label]}.jpg"))

        # Also save the associated annotations to a csv file
        shutil.copy(best_detection_csv, os.path.join(best_folder, f"class_{class_labels_mapping[class_label]}.csv"))

    # Save the processed frames as a video
    processed_clip = ImageSequenceClip(processed_frames, fps=video.fps)
    processed_clip = processed_clip.set_audio(video.audio)
    processed_clip.write_videofile(os.path.join(video_output_folder, f"{video_filename_base}_processed.mp4"), codec='libx264', audio_codec='aac')

    # Save original video here as well 
    video.write_videofile(os.path.join(video_output_folder, f"{video_filename_base}_original.mp4"), codec='libx264', audio_codec='aac')


    return

if __name__ == "__main__": 
    editor_data_folder = './video_editing_data'
    weights_file = 'logone/yolov8n-finetune-round-one2/weights/best.pt'
    video_file = 'media/MVI_1043.MP4'
    class_reference_file = 'logone/data/openlogo/yolo_finetune_split/data.yaml'

    args = {
        'weights_file': weights_file,
        'video_file': video_file,
        'output_folder': editor_data_folder,
        'class_reference_file' : class_reference_file
    }

    main(args) 