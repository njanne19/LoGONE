import fiftyone as fo 
import os, pathlib
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Initialize and empty FiftyOne dataset 
dataset_name = "OpenLogo"
dataset_already_loaded = False 
if fo.dataset_exists(dataset_name):
    dataset = fo.load_dataset(dataset_name)
    dataset_already_loaded = True

if not dataset_already_loaded:
    dataset = fo.Dataset(dataset_name) 

    # Your data'sets root directory 
    root_dir = "../data/openlogo" 


    def parse_voc_xml(xml_path): 
        """
        Parse a PASCAL VOC XML file into fiftyone format. 
        See more at https://docs.cvat.ai/docs/manual/advanced/formats/format-voc/
        """

        tree = ET.parse(xml_path) 
        root = tree.getroot() 

        width= int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        depth = int(root.find('size').find('depth').text)

        boxes = [] 
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox') 
            xmin = float(bndbox.find('xmin').text) / width
            ymin = float(bndbox.find('ymin').text) / height
            xmax = float(bndbox.find('xmax').text) / width
            ymax = float(bndbox.find('ymax').text) / height

            # Create a fifty one bounding box [xmin, ymin, width, height]
            box = [xmin, ymin, xmax - xmin, ymax - ymin]

            # Create a fifty one label 
            label = fo.Detection(label=name, bounding_box=box) 

            boxes.append(label)

        # Directly return an ImageMetadataInstance 
        metadata = fo.ImageMetadata(width=width, height=height, depth=depth)

        return boxes, metadata 


    # Iterate over all the images in the dataset, parse the XML, and add to fiftyOne 
    print("Loading dataset...")

    image_folder = 'JPEGImages'
    annotations_folder = 'Annotations' 

    files_with_progress = tqdm(os.listdir(os.path.join(root_dir, image_folder)))
                               
    for file in files_with_progress:
        files_with_progress.set_description(f"Processing {file:<30.30}...")
        image_path = os.path.join(root_dir, image_folder, file)
        image_path = os.path.abspath(image_path)

        # Corresponding XML file for annotations
        annotation_path = image_path.replace(image_folder, annotations_folder).replace(".jpg", ".xml")

        # Initialize sample
        sample = fo.Sample(filepath=image_path)

        if os.path.exists(annotation_path):
            # Parse the VOC XML file
            detections, img_metadata = parse_voc_xml(annotation_path)

            # Add detections and metadata to sample
            sample["ground_truth"] = fo.Detections(detections=detections)
            sample["metadata"] = img_metadata

        # Add sample to dataset
        dataset.add_sample(sample)

    
    # Now that we have added all the samples, we need to go back in and assign training/testing labels from the 
    # openlogo/ImageSets/class_sep folder. In this folder, there is a _train.txt and a _test.txt for each class. 
    sets_folder = 'ImageSets/Main/train_test'
    categories_with_progress = tqdm(os.listdir(os.path.join(root_dir, sets_folder)))

    print(f"Getting training/test labels...")
    train_test_lookup = {}

    for category in categories_with_progress:
        categories_with_progress.set_description(f"Processing {category:<30.30}...")
        category_path = os.path.join(root_dir, sets_folder, category)

        # Check if this is a file 
        if os.path.isfile(category_path):
            with open(category_path, 'r') as f:
                for line in f:
                    filename = line.strip()
                    train_test_lookup[filename] = "train" if category.startswith("train") else "test"

    # Now, assign labels in single pass 
    print(f"Assigining training/test labels...")

    elements_with_progress = tqdm(dataset)

    for sample in elements_with_progress:
        elements_with_progress.set_description(f"Processing {sample.filepath:<30.30}...")
        filename = pathlib.Path(sample.filepath).stem
        status = train_test_lookup.get(filename, None)

        if filename in train_test_lookup: 
            sample.tags.append("default_split/" + train_test_lookup[filename])
            sample.save() 



session = fo.launch_app(dataset)

print("FiftyOne app launched. Press Ctrl+C to exit.")
try: 
    # This loop runs forever 
    while True: 
        pass 
except KeyboardInterrupt: 
    print("FiftyOne app closed.")
