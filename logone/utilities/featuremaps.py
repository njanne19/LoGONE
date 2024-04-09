import torch 
import torch.nn 
import matplotlib.pyplot as plt 



def high_activation_sampling(feature_maps, num_samples=64): 
    """
    Selects feature maps (channels) of a convolutional layer output with the 
    highest mean actiation values. 


    Parameters: 
    - feature_maps (torch.Tensor), size = (num_filters, height, widht) : The output of a convolutional layer
    - num_samples (int): The number of feature maps to select.

    Returns: 
    - Indicies of the feature maps with the highest activation. 
    """

    # Compute the mean activation for each feature map 
    mean_activations = feature_maps.mean(dim=(1, 2))

    # Select the indices of the top 'num_samples' feature_maps 
    num_samples = min(num_samples, feature_maps.shape[0]) 
    indices = torch.argsort(mean_activations)[-num_samples:]

    return indices


def visualize_layers_and_input(input_image, layer_outputs, activations_per_layer = [2, 4, 8, 16], sampling='high_activation'): 
    """
    Visualize the input image and selected feature maps from each convolutional layer. 

    Parameters: 
    - input_image (torch.Tensor), size = (3, height, width): The input image to the model
    - layer_outputs (list of torch.Tensor): The output of each convolutional layer in the model, 
    all of size (num_filters, height, width) (num filters varies by layer)
    - activations_per_layer (int): a list indicating how many activations to display from each layer. 
    """

    # De-batch layer outputs 
    if layer_outputs[0].dim() == 4: 
        input_image = input_image.squeeze(0)
        layer_outputs = [layer.squeeze(0) for layer in layer_outputs]

    # Total columns = 1 (for the input image) + number of layers 
    total_cols = 1 + len(layer_outputs)
    # The number of rows is determined by the maximum number of activations to display from any layer
    total_rows = max(activations_per_layer)   

    fig = plt.figure(layout='constrained', figsize=(total_cols * 4, total_rows * 4))

    # Display the input image in the first column, across all rows 
    subfigs_column = fig.subfigures(1, total_cols, wspace=0.05, hspace=0.05)

    for layer_index, column in enumerate(subfigs_column): 
        
        # Make one subplot per # of activations to display 
        if layer_index == 0: 
            row_axs = [column.subplots(1, 1)]
        else: 
            row_axs = column.subplots(activations_per_layer[layer_index-1], 1)

        # Calculate the indices to display 
        if sampling == 'high_activation':
            indices = high_activation_sampling(layer_outputs[layer_index - 1], num_samples=activations_per_layer[layer_index - 1])
        else: 
            raise ValueError(f"Invalid sampling method: {sampling}")
        
        # Display the selected feature maps
        for row_index, row_ax in enumerate(row_axs): 
            ax = row_ax 
            # ax.axis('off')

            if layer_index == 0: 
                # Display the input image 
                ax.imshow(input_image.permute(1, 2, 0))
                ax.set_title("Input Image")
            else:
                # Display the selected feature maps 
                map_to_display = indices[row_index]
                ax.imshow(layer_outputs[layer_index - 1][map_to_display].detach().cpu())
                ax.set_title(f"Filter {row_index}")

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(12)

        if layer_index != 0: 
            column.suptitle(f"Layer {layer_index}")
        else: 
            column.suptitle("Input Image")


    fig.suptitle("Input Image and Selected Feature Maps")
            

    plt.show()
