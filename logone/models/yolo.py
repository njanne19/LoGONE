from ultralytics import YOLO 
import os
import urllib.request

YOLO_MODEL_DICT = {
    "yolov8n" : "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt", 
    "yolov8s" : "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt",
    "yolov8m" : "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt",
    "yolov8l" : "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt",
    "yolov8x" : "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt",
}

def download_file(url, filepath): 
    if not os.path.exists(filepath): 
        print(f"Downloading {url} to {filepath}...")
        with urllib.request.urlopen(url) as response: 
            # Read the data from the url 
            data = response.read()

            # Open a local file in binary write mode 
            with open(filepath, 'wb') as f: 
                f.write(data)
    else: 
        print(f"{filepath} already exists. Skipping download...")

# Load a model 
model = "yolov8n"


# Check to see if a weights folder for this model 
current_parent_directory = os.path.abspath(os.path.join(__file__, os.pardir))
weights_folder = os.path.join(current_parent_directory, "weights")
model_folder = os.path.join(weights_folder, "yolo")

os.makedirs(model_folder, exist_ok=True)
os.makedirs(weights_folder, exist_ok=True)

# Download the model weights
model_url = YOLO_MODEL_DICT[model]
model_filepath = os.path.join(model_folder, f"{model}.pt")
download_file(model_url, model_filepath)

# Load the model 
model = YOLO(model_filepath)

# Get the dataset folder 
dataset_file = 'logone/data/openlogo/yolo_split1/data.yaml'

# Train the model 
results = model.train(
    data=dataset_file, 
    epochs=100, 
    imgsz=640, 
    project='logone',
    batch=-1,
    save_period=25,
    cache = True,
    name='yolov8n-allclass',)

# Evalute the model 
metrics = model.val() 
