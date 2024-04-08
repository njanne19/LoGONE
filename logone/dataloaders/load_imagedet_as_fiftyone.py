import fiftyone as fo 
import os 
import xml.etree.ElementTree as ET

# Initialize and empty FiftyOne dataset 
dataset_name = "LogoDet"
dataset_already_loaded = False 
if fo.dataset_exists(dataset_name):
    dataset = fo.load_dataset(dataset_name)
    dataset_already_loaded = True

if not dataset_already_loaded:
    dataset = fo.Dataset("LogoDet") 

    # Your data'sets root directory 
    root_dir = "../data/LogoDet-3K" 


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

    for category in os.listdir(root_dir): 
        # First get cateogry path
        category_path = os.path.join(root_dir, category)

        # Check if this is a directory
        if os.path.isdir(category_path): 
            print(f"Loading category: {category}")

            # Then get subcategory 
            for subcategory in os.listdir(category_path): 
                subcategory_path = os.path.join(category_path, subcategory)

                # Check if this is a directory
                if os.path.isdir(subcategory_path): 
                    print(f"Loading subcategory: {subcategory}")
                    # Iterate over the items in the subcategory 
                    for item in os.listdir(subcategory_path): 
                        if item.endswith(".jpg"): 
                            print(f"Loading item: {item}")
                            image_path = os.path.join(subcategory_path, item)

                            # Convert to absolute path 
                            image_path = os.path.abspath(image_path)

                            # Corresponding XML file for annotations 
                            annotation_path = image_path.replace(".jpg", ".xml") 

                            # Initialize sample 
                            sample = fo.Sample(filepath=image_path)

                            # Custom metadata for category and subcategory 
                            sample["category"] = category
                            sample["subcategory"] = subcategory

                            if os.path.exists(annotation_path): 
                                # Parse the VOC XML file 
                                detections, img_metadata = parse_voc_xml(annotation_path)

                                # Add detections and metadata to sample 
                                sample["ground_truth"] = fo.Detections(detections=detections)
                                sample["metadata"] = img_metadata

                            # Add sample to dataset
                            dataset.add_sample(sample)

session = fo.launch_app(dataset)

print("FiftyOne app launched. Press Ctrl+C to exit.")
try: 
    # This loop runs forever 
    while True: 
        pass 
except KeyboardInterrupt: 
    print("FiftyOne app closed.")