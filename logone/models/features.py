import torch 
import torch.nn as nn
import torchvision 
from torchsummary import summary


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

    summary(model, input_size=(3, 224, 224), device='cpu') 

    model.eval() 
    input_tensor = torch.randn(1, 3, 224, 224)
    features = model(input_tensor)