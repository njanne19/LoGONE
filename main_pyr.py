import os
import torch
import torch.nn as nn
from logone.dataloaders.logo_transform_dataset import LogoTransformDataset
from logone.models.logo_transform import PyramidCNN, UNet
from torch.utils.data import DataLoader

if __name__ == "__main__":
    x = torch.rand((256,256,6))
    h, w, in_channels = x.shape
    out_classes = 9

    mode = "sm" 

    match mode:
        case "sm":
            model_weight_path = os.path.join(os.getcwd(), "logone", 'model_weights', 'weights0.pth')
            k_size = 3
        case "med":
            model_weight_path = os.path.join(os.getcwd(), "logone", 'model_weights', 'weights1.pth')
            k_size = 5
        case "lg":
            model_weight_path = os.path.join(os.getcwd(), "logone", 'model_weights', 'weights2.pth')
            k_size = 7

    model = PyramidCNN(in_channels, out_classes, h, w, kernel_size=k_size)

    # model_weight_path = os.path.join(os.getcwd(), "logone", 'model_weights', 'weights1.pth')
    model_weight_path = None
    if model_weight_path is None:
        model_weight_dir = os.path.join(os.getcwd(), 'logone', 'model_weights')

        ui = 0
        model_weight_path = os.path.join(model_weight_dir, 'weights' + str(ui) + '.pth')
        while os.path.exists(model_weight_path):
            ui += 1
            model_weight_path = os.path.join(model_weight_dir, 'weights' + str(ui) + '.pth')

    else:
        model.load_state_dict(torch.load(model_weight_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_data = LogoTransformDataset(os.path.join(os.getcwd(), 'logone', 'utilities', 'labels_train.csv'), 
                                         os.path.join(os.getcwd(), 'logone', 'utilities', 'transformed_256'))
    val_data = LogoTransformDataset(os.path.join(os.getcwd(), 'logone', 'utilities', 'labels_test.csv'),
                                     os.path.join(os.getcwd(), 'logone', 'utilities', 'transformed_256'))
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=590, shuffle=True)

    print('Device: ', device)

    num_epochs = 10

    loss_norm = len(val_data)//64 + 1
    # Train the model

    training_loss_history = []
    validation_loss_history = []
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