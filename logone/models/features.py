import torch 
import torch.nn as nn
import torchvision 
from torchvision.transforms import ToTensor
from torchsummary import summary
from logone.utilities.featuremaps import visualize_layers_and_input
import fiftyone as fo 
from PIL import Image


class ResNet50Features(nn.Module):
    def __init__(self):
        super(ResNet50Features, self).__init__()
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.convolutional_features = nn.Sequential(*list(resnet.children())[:-2])

        # Manually select the layers to be used as features 
        self.selected_layers = [4, 5, 6, 7]
        self.layer_names = {4: 'End of Conv2_x', 5: 'End of Conv3_x', 6: 'End of Conv4_x', 7: 'End of Conv5_x'}

        
    def forward(self, x):
        selected_features = []
        for i, layer in enumerate(self.convolutional_features):
            x = layer(x)
            if i in self.selected_layers:
                selected_features.append(x)
                print(f"Extracted Layer: {self.layer_names[i]}, Output Shape: {x.shape}")
        return selected_features
    


if __name__ == "__main__": 
    model = ResNet50Features()
    print(model) 

    # Show the feature map. 
    # Try to load one of the current datasets: 
    try: 
        dataset = fo.load_dataset('OpenLogo') 
    except: 
        try: 
            dataset = fo.load_dataset('LogoDet') 
        except: 
            print("Please load a dataset to visualize the feature maps.")
            dataset = None

    input_tensor = None

    if dataset is not None:  
        train_dataset = dataset.match({"split": "train"})

        # Get all sample IDs from the filtered dataset 
        train_sample_ids = train_dataset.values("id")

        # Randomly select one sample ID 
        random_sample_id = train_sample_ids[0]

        # Load the sample 
        random_sample = train_dataset[random_sample_id]

        # Load the image
        image_path = random_sample["filepath"]
        image = Image.open(image_path)

        # Preprocess the image
        transform = ToTensor()
        input_tensor = transform(image).unsqueeze(0)

    

    if input_tensor is None: 
        input_tensor = torch.rand(1, 3, 224, 224)

    model.eval() 
    features = model(input_tensor)
    visualize_layers_and_input(input_tensor, features)