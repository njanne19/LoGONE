import os
import torch
import torch.nn as nn
import numpy as np
from logone.dataloaders.logo_transform_dataset import LogoTransformDataset
from logone.models.logo_transform import PyramidCNN, UNet
from torch.utils.data import DataLoader
from logone.utilities.image_transformer import apply_logo_transform
import matplotlib.pyplot as plt

def validationloader():
    data_dir = os.path.join(os.getcwd(), 'logone', 'utilities')
    stat_path = os.path.join(data_dir, 'data_norm_stats.csv')
    val_data = LogoTransformDataset(os.path.join(data_dir, 'labels_test.csv'),
                                    os.path.join(data_dir, 'transformed_256'),
                                    stat_path=stat_path)
    return DataLoader(val_data, batch_size=590, shuffle=True)

def load_plot_model(mode):
    model = load_model(mode)[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(os.getcwd(), 'logone', 'utilities')
    stat_path = os.path.join(data_dir, 'data_norm_stats.csv')

    val_data = LogoTransformDataset(os.path.join(data_dir, 'labels_train.csv'),
                                    os.path.join(data_dir, 'transformed_256'),
                                    stat_path=stat_path)
    val_dataloader = DataLoader(val_data, batch_size=590, shuffle=True)

    norm_stats = np.loadtxt(stat_path)
    mean_img = norm_stats[0]
    std_img = norm_stats[1]
    mean_label = norm_stats[2]
    std_label = norm_stats[3]

    for data in val_dataloader:
        norm_img_stacks, norm_labels = data[0].to(device), data[1].to(device)
        norm_pred_labels = model(norm_img_stacks)
        pred_labels = norm_pred_labels*std_label + mean_label
        labels = norm_labels*std_label + mean_label
        img_stacks = norm_img_stacks*std_img + mean_img

        for i in range(img_stacks.shape[0]):
            img_stack = img_stacks[i,:,:,:]
            label = labels[i,:].detach().numpy()
            pred_label = pred_labels[i,:].detach().numpy()
            orig_img = img_stack[:,:,:3].numpy()
            orig_img_transformed = img_stack[:,:,3:].numpy()
            pred_img_transformed = apply_logo_transform(orig_img, *pred_label)
            # pred_img_transformed = apply_logo_transform(orig_img, *label)

            # print('GT Label: ', label)
            # print('Predicted Label: ', pred_label)
            print('Error: ', pred_label-label)
            print('Transformed Img Size: ', pred_img_transformed.shape)

            fig0, ax0 = plt.subplots()
            ax0.imshow(orig_img.astype(np.uint8))
            ax0.set_title('Original Logo')

            fig1, ax1 = plt.subplots()
            ax1.imshow(pred_img_transformed.astype(np.uint8))
            ax1.set_title("Predicted Transform")

            fig2, ax2 = plt.subplots()
            ax2.imshow(orig_img_transformed.astype(np.uint8))
            ax2.set_title("Ground Truth Transformation")

            plt.show()


def load_model(mode):
    match mode:
        case "sm":
            model_weight_path = os.path.join(os.getcwd(), "logone", 'model_weights', 'cylweights0.pth')
            k_size = 3
        case "med":
            model_weight_path = os.path.join(os.getcwd(), "logone", 'model_weights', 'cylweights1.pth')
            k_size = 5
        case "lg":
            model_weight_path = os.path.join(os.getcwd(), "logone", 'model_weights', 'cylweights2.pth')
            k_size = 7

    in_channels = 6
    out_classes=9
    # h=256
    # w=256
    h=128
    w=128
    model = PyramidCNN(in_channels, out_classes, h, w, kernel_size=k_size)
    model.load_state_dict(torch.load(model_weight_path))
    return model, model_weight_path

def train_model(mode, load=False):
    if load:
        model, model_weight_path = load_model(mode)
    else:
        model_weight_dir = os.path.join(os.getcwd(), 'logone', 'model_weights')

        match mode:
            case "sm": 
                ui = 0
                k_size = 3
            case "med":
                ui = 1
                k_size=5
            case "lg":
                ui = 2
                k_size=7

        model_weight_path = os.path.join(model_weight_dir, 'cylweights' + str(ui) + '.pth')
        in_channels = 6
        out_classes=9
        # h=256
        # w=256
        h=128
        w=128
        model = PyramidCNN(in_channels, out_classes, h, w, kernel_size=k_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    data_path = os.path.join(os.getcwd(), 'logone', 'utilities')
    stat_path = os.path.join(data_path, 'data_norm_stats.csv')
    train_path = os.path.join(data_path, 'labels_train.csv')
    test_path = os.path.join(data_path, 'labels_test.csv')
    img_dir = os.path.join(data_path, 'transformed_256')

    train_data = LogoTransformDataset(train_path, img_dir, stat_path)
    val_data = LogoTransformDataset(test_path, img_dir, stat_path)

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=590, shuffle=True)

    num_epochs = 10
    # Train the model
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if i == 5: break
            # if i % 2000 == 1999:  # print every 2000 mini-batches
        print('[%d] loss: %.3f' % (epoch + 1, running_loss/(i+1)))
        with open(os.path.join(os.getcwd(), 'training_loss.csv'), "a") as f:
            f.write('Epoch ' + str(epoch) + ':   Training Loss=' + str(running_loss))
            # running_loss = 0.0
        val_data = next(iter(val_dataloader))
        val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
        val_out = model(val_inputs)
        val_loss = criterion(val_out, val_labels)
        with open(os.path.join(os.getcwd(), 'validation_loss.csv'), 'a') as f:
            f.write('Epoch ' + str(epoch) + ': Validation Loss=' + str(val_loss))
        print('Validation loss: ', val_loss.item())

    print('Finished Training')
    torch.save(model.state_dict(), model_weight_path)

if __name__ == "__main__":
    # load_plot_model("lg")
    train_model('lg', load=False)