import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom

import pdb

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython import display
import time


class NpzDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.file_list = [f for f in os.listdir(dir_path) if f.endswith('.npz')]
        self.dir_path = dir_path
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.dir_path, self.file_list[idx])
        list = load_arrays(file_path)
        inputs = list[0]
        outputs = list[1]
       
        input_tensor = torch.from_numpy(inputs)
        output_tensor = torch.from_numpy(outputs)
        
        if self.transform:
            input_tensor = self.transform(input_tensor)
            output_tensor = self.transform(output_tensor)

        return input_tensor, output_tensor
    
def n_0_and_1(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    denominator = arr_max - arr_min
    if denominator <= 0.01:
        return arr/(arr_max+1e-6)
    return (arr - arr_min) / (denominator)

def n_neg1_and_1(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    denominator = arr_max - arr_min
    if denominator <= 0.01:
        return arr/(arr_max+1e-6)
    return (2*(arr - arr_min) / (denominator + 1e-6)) - 1   
    
def load_arrays(path):
    
    #input data
    data = np.load(path)
    pre_fl = n_0_and_1(data['array1'])
    pre_dob = n_0_and_1(data['array2'])

    #n arrays
    #precipitation has the largest resolution so it is probably the most likely to have this issue, but I did v comp and u comp as well
    
        
    v_comp = data['array3']
    u_comp = data['array4']
    precip = data['array5']
    cover = data['array6']

    v_comp = v_comp.astype(float)
    u_comp = u_comp.astype(float)
    precip = precip.astype(float)
   

  #  print('\nvcomp')
    v_comp = n_neg1_and_1(v_comp) if len(v_comp.shape) != 3 else  n_neg1_and_1(np.squeeze(v_comp))
   # print('\nuvomp')
    u_comp = n_neg1_and_1(u_comp) if len(u_comp.shape) != 3 else n_neg1_and_1(np.squeeze(u_comp))
   # print(precip.shape, v_comp.shape, u_comp.shape)
    
  
   

    
    if len(precip.shape) == 3:
        precip = n_0_and_1(precip.reshape(precip.shape[:-1]))
    else:
    #    print("This is useful")
        precip = n_0_and_1(precip)

    
        
    #elif len(precip.shape) == 1:
    #    precip = n_0_and_1(precip[:,np.newaxis]) 
    #else: 
    #    precip = n_0_and_1(precip)

    #print(precip.shape, v_comp.shape, u_comp.shape)
    cover = cover.astype(float) 
   # print(cover.shape)
    cover =  n_0_and_1(cover) if len(cover.shape) !=3 else n_0_and_1(np.squeeze(cover))
   
    
   

    # post_fl = data['array7']
    post_dob = n_0_and_1(data['array8'])

    # Determine the target shape
    target_shape = pre_fl.shape  # or any other shape you want

    
    #print(post_dob.shape == target_shape)
    # Resize arrays
    
    v_comp = zoom(v_comp, (target_shape[0] / v_comp.shape[0], target_shape[1] / v_comp.shape[1]))
    u_comp = zoom(u_comp, (target_shape[0] / u_comp.shape[0], target_shape[1] / u_comp.shape[1]))

    precip = zoom(precip, (target_shape[0] / precip.shape[0], target_shape[1] / precip.shape[1]))

    cover = zoom(cover, (target_shape[0] / cover.shape[0], target_shape[1] / cover.shape[1]))
    inputs = np.stack((pre_fl, pre_dob, v_comp, u_comp, precip, cover), axis=0)
    return [inputs, post_dob]
    
def collate_fn(batch):
    # Separate sequences and labels
    sequences, labels = zip(*batch)

    # Get the maximum dimensions in the batch
    max_dim1 = max([seq.shape[1] for seq in sequences])
    max_dim2 = max([seq.shape[2] for seq in sequences])
    print(f'max dimension 2: {max_dim1}, max dimension 3: {max_dim2}')
    
    
  

    # Pad sequences to match the maximum dimensions
    sequences_padded = [torch.nn.functional.pad(seq, (0, max_dim2-seq.shape[2]+1, 0, max_dim1-seq.shape[1]+1, 0, 0)) for seq in sequences]
    labels_padded = [torch.nn.functional.pad(label,(0, max_dim2-label.shape[1], 0, max_dim1 - label.shape[0])) for label in labels]

    return torch.stack(sequences_padded), torch.stack(labels_padded)

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        #could change other factors about this such as dialation and the stride
        self.encoder_conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.encoder_conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        
        self.decoder_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder_conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_conv4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        

    def forward(self, x):
        # Encoder
        x = F.relu(self.encoder_conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.encoder_conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.encoder_conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.encoder_conv4(x))
        

        # Decoder
            #This doubles the scale through interpolation
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.decoder_conv1(x))

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.decoder_conv2(x))

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.decoder_conv3(x))

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.sigmoid(self.decoder_conv4(x))
        return x

def main():
    root = '/Users/alexthomas/Desktop/Pytorch/data/train/'
    data = NpzDataset(root)

    print(f"Number of items: {len(data)}")

    # Create data loaders
    print('loading model and Dataloader: ')
    data_loader = DataLoader(data, batch_size=32, collate_fn=collate_fn)
    


    # See comment below about epoch for loop
    # checkpoint = torch.load("checkpoints/checkpoint_0.pth")

   # device = "cuda"

    model = FCN()
    # model.load_state_dict(checkpoint["model_state_dict"])
    model.train()
   # model.to(device)

    # Define a loss function and optimizer
    print("defining optimizer:")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Load the optimizer state
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Train the model
    num_epochs = 200

    # Create inf loss
    best_epoch_loss = np.inf #min(sum(checkpoint["loss"]), np.inf)

    for epoch in range(num_epochs):
        losses = []
        epoch_loss = 0.0
        total_batches = len(data_loader)

        for batch_num, wrapped in enumerate(data_loader):

            inputs, targets = wrapped[0], wrapped[1]
            # Move data to the appropriate device if necessary
            
            #inputs = inputs.to(device)
            #targets = targets.to(device)

            # Zero the parameter gradients
            t1 = time.time()
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs.float())

            # Detach to free up GPU mem
            outputs.detach()
            inputs.detach()
            targets.detach()

            outputs = outputs.to("cpu")
            inputs = inputs.to("cpu")
            targets = targets.to("cpu")

            outputs = outputs.squeeze(1)

            if outputs.shape != targets.shape:

                # handle case for first dimension
                if outputs.shape[1] < targets.shape[1]:
                    outputs = torch.nn.functional.pad(
                        outputs, (0, 0, 0, targets.shape[1] - outputs.shape[1], 0, 0)
                    )
                if outputs.shape[1] > targets.shape[1]:
                    outputs = outputs[:, : targets.shape[1], :]

                # handle case for second dimension
                if outputs.shape[2] < targets.shape[2]:
                    outputs = torch.nn.functional.pad(
                        outputs, (0, targets.shape[2] - outputs.shape[2], 0, 0, 0, 0)
                    )

                if outputs.shape[2] > targets.shape[2]:
                    outputs = outputs[:, :, : targets.shape[2]]

            # Calculate loss
            loss = criterion(outputs, targets.float())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            t2 = time.time()
            percent = (batch_num / total_batches) * 100

            keep_text = (
                f"Epoch Number: {epoch} ---- Batch: {batch_num} ---- Percentage done: {percent:.2f}% --- Time: {t2-t1}"
            )
            
            # Detach
            loss.detach()
            loss = loss.to("cpu").item()
            epoch_loss += loss

            if batch_num % 500 == 0:
                display.clear_output()
                display.display(keep_text)
                for param_group in optimizer.param_groups:
                    display.display(f"Learning rate: {param_group['lr']}")
                    display.display("Batch Loss", loss)

                    display.display("Epoch Loss:", epoch_loss)

            losses.append(loss)

        if epoch_loss < best_epoch_loss:
            print(f"Model saved as checkpoint_{epoch}.pth for epoch {epoch}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": losses,
                },
                f"checkpoints/checkpoint_{epoch}.pth",
            )
            best_epoch_loss = epoch_loss

    
if __name__ == '__main__':
    main()