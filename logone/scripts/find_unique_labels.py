import fiftyone as fo 
from PIL import Image, ImageDraw, ImageFont

def create_collage(first_instances, width, init_height, num_cols): 
    num_images = len(first_instances) 
    num_rows = (num_images + num_cols - 1) // num_cols
    thumbnail_width = width // num_cols 
    max_height = init_height // num_rows 
    
    # Create a new blank image 
    collage = Image.new('RGB', (width, init_height), 'white')
    draw = ImageDraw.Draw(collage) 
    
    x_offset = 0 
    y_offset = 0 
    current_col = 0 
    
    for label, filepath in first_instances.items(): 
        try: 
            img = Image.open(filepath) 
            img.thumbnail((thumbnail_width, max_height))
            
            # Calculate the position 
            if current_col >= num_cols: 
                y_offset += max_height + 20 
                x_offset = 0 
                current_col = 0 
            
            # Paste image 
            collage.paste(img, (x_offset, y_offset + 20))
            
            # Add label below the image 
            draw.text((x_offset, y_offset), label, fill="black") 
            
            x_offset += thumbnail_width 
            current_col += 1
            
        except Exception as e: 
            print(f"Failed to process image {filepath}: {e}")
            
    return collage

def main(): 
    
    # Load the dataset 
    try: 
        dataset = fo.load_dataset("OpenLogo") 
    except: 
        print("Dataset not found. Please load the dataset first with FityOne.")
        return
    
    # Dictionary to store the unique labels 
    first_instances = {} 
    
    # Iterate over all samples in the dataset
    print(f"Finding unique labels among {len(dataset)} samples...")
    for sample in dataset: 
        # Access detections in the 'ground_truth' field 
        try:
            detections = sample["ground_truth"].detections 
        except: 
            continue
        
        # Check each detection for its label 
        for detection in detections: 
            label = detection.label
            
            # If the label is not in the dictionary, add it
            if label not in first_instances: 
                print(f"Found new label: {label}")
                first_instances[label] = sample.filepath 
                
    
    # Generate the results in a PDF, where we have the label and the first instance 
    print("Generating collage with unique labels...")
    collage_width = 10000 
    collage_height = 13000 
    columns = 10
    
    collage_image = create_collage(first_instances, collage_width, collage_height, columns)
    collage_image.save('label_collage.jpg') 
    
    return 


if __name__ == "__main__": 
    main()