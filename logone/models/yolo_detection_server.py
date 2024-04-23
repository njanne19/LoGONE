from ultralytics import YOLO 
import os 
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import shutil
from moviepy.editor import VideoFileClip, ImageSequenceClip
from tqdm import tqdm
import base64


app = Flask(__name__) 
CORS(app) 

# Define YOLO model (to be used later) 
print("Loading YOLO model...")
model = YOLO('logone/yolov8n-finetune-round-one2/weights/best.pt')

# Maintian global variables keeping track of the server records 
SERVER_VIDEO_SAVE_PATH = "server_data"
RECORDS_TABLE_FILEPATH = os.path.join(SERVER_VIDEO_SAVE_PATH, "records.csv")
RECORDS_TABLE = None 


def load_or_create_record_table(file_path):  
    if os.path.exists(file_path): 
        df = pd.read_csv(file_path) 
    else: 
        # Create an empty DataFrame with the specific columns 
        df = pd.DataFrame(columns=['video_filename', 'subdirectory'])

        # Save the fataframe to a CSV file 
        df.to_csv(file_path, index=False) 

    return df 

def add_record_table_entry(video_filename, subdirectory): 
    global RECORDS_TABLE
    # Check to see if the records table is none, if it is, create it first 
    if RECORDS_TABLE is None:
        RECORDS_TABLE = load_or_create_record_table(RECORDS_TABLE_FILEPATH)

    # Creat a new entry as a DataFrame row 
    new_entry = pd.DataFrame({
        'video_filename': [video_filename],
        'subdirectory': [subdirectory]
    })

    # Append the new entry to the existing DataFrame 
    RECORDS_TABLE = pd.concat([RECORDS_TABLE, new_entry], ignore_index=True)

    # save the updated DataFrame to the CSV file
    RECORDS_TABLE.to_csv(RECORDS_TABLE_FILEPATH, index=False)



def remove_record_table_entry(video_filename):
    global RECORDS_TABLE
    if RECORDS_TABLE is None: 
       RECORDS_TABLE = load_or_create_record_table(RECORDS_TABLE_FILEPATH)

    # Find the row that matches the video_filename 
    mask = RECORDS_TABLE['video_filename'] == video_filename

    # Remove the row from the DataFrame 
    RECORDS_TABLE = RECORDS_TABLE[~mask]

    # Save the updated DataFrame to the CSV file 
    RECORDS_TABLE.to_csv(RECORDS_TABLE_FILEPATH, index=False)


@app.route('/yolo', methods=['POST']) 
def handle_video(): 
    # Check if the post request has the file part 
    if 'video' not in request.files: 
        return jsonify({'error': 'No file part'}), 400
    file = request.files['video'] 
    if file.filename == '': 
        return jsonify({'error': 'No selected file'}), 400

    if file: 
        filename = file.filename

        print(f"Received video file: {filename}")

        # Get the filename without the extension for subdir 
        subdir = os.path.splitext(filename)[0]

        # Check to see if that subdirectory already exists, if it does, delete it and its content 
        subdir_path = os.path.join(SERVER_VIDEO_SAVE_PATH, subdir)
        if os.path.exists(subdir_path): 
            shutil.rmtree(subdir_path)
            remove_record_table_entry(filename)

        # Add new video to the record table
        print(f"Adding {filename} to the records table...")
        add_record_table_entry(filename, subdir)

        # Create the subdirectory
        os.makedirs(subdir_path, exist_ok=True)

        # Save the video to the subdirectory 
        video_path = os.path.join(subdir_path, filename)
        file.save(video_path)

        # Process video with yolo 
        print(f"Processing video with YOLO...")
        modified_video_path = process_video_with_yolo(subdir_path, filename) 

        # Then convert to base64 and return back to the client 
        base64_string = convert_file_to_base64(modified_video_path)
        return jsonify({'message': 'Video processed', 'base64Video': base64_string}), 200


def process_video_with_yolo(subdir_path, filename): 

    # Get the current video path 
    video_path = os.path.join(subdir_path, filename)
    
    # Generate the modified video path 
    modified_video_path = os.path.join(subdir_path, (os.path.splitext(filename)[0] + "_yolo.mp4"))

    # Load the video 
    clip = VideoFileClip(video_path)

    # Process each frame 
    processed_frames = [process_frame_with_yolo(frame) for frame in tqdm(clip.iter_frames())]

    # Then generate a new video clip 
    processed_clip = ImageSequenceClip(processed_frames, fps=clip.fps)
    processed_clip = processed_clip.set_audio(clip.audio)
    processed_clip.write_videofile(modified_video_path, codec='libx264', audio_codec='aac')

    return modified_video_path


def process_frame_with_yolo(frame):  

    # First get the BGR image from the RGB 
    bgr_frame = frame[:, : ,::-1]

    # Then pass through the yolo model 
    results = model(bgr_frame, agnostic_nms=True, augment=True) 
    annotated_frame = results[0].plot() 

    # Then get the RGB image 
    rgb_frame = annotated_frame[:, : ,::-1]

    return rgb_frame


def convert_file_to_base64(file_path): 
    try: 
        # Read the file in binary mode 
        with open(file_path, "rb") as video_file: 
            encoded_string = base64.b64encode(video_file.read())
            base64_string = encoded_string.decode('utf-8')
            return base64_string
    except: 
        return None
    
if __name__ == "__main__": 
    app.run(host='0.0.0.0', port=5000, debug=True)