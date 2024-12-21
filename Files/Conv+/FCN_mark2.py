

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import functional as TF
import random
import time
from IPython import display

class NpzDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.file_list = [f for f in os.listdir(dir_path) if f.endswith('.npz')]
        self.dir_path = dir_path
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.dir_path, self.file_list[idx])
        file = np.load(file_path)
        inputs = file['inputs']
        outputs = file['output']

        input_tensor = torch.from_numpy(inputs).float()
        output_tensor = torch.from_numpy(outputs).float()
        
        seed = random.randint(0, 2 ** 32)
        
        input_tensor, output_tensor = self.random_crop_with_seed(input_tensor, output_tensor, (64, 64), seed)   
        input_tensor, output_tensor = self.random_flip_with_seed(input_tensor, output_tensor, seed)
        
        return input_tensor, output_tensor


    def random_crop_with_seed(self, input_tensor, output_tensor, output_size, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)  # If using numpy for random operations

        height, width = input_tensor.shape[-2:]  # Assuming input_tensor is of shape [C, H, W] or [H, W]
        crop_height, crop_width = output_size

        if height < crop_height or width < crop_width:
            raise ValueError("Crop size must be smaller than the dimensions of the input tensor.")

        y = random.randint(0, height - crop_height)
        x = random.randint(0, width - crop_width)

        cropped_input = input_tensor[..., y:y+crop_height, x:x+crop_width]
        cropped_output = output_tensor[..., y:y+crop_height, x:x+crop_width]

        return cropped_input, cropped_output


    def random_flip_with_seed(self, image, output, seed):
        random.seed(seed)

        # Decide whether to flip or not
        if random.random() > 0.5:
            image = TF.hflip(image)
            output = TF.hflip(output)
        if random.random()>0.5:
            image = TF.vflip(image)
            output = TF.vflip(output)

        return image, output


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.encoder_conv1 = nn.Conv2d(6, 128, kernel_size=5, padding='same')
        self.encoder_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.encoder_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        
        #self.attention = SpatialAttention()
    
        self.decoder_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding='same')
        self.decoder_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding='same')
        self.decoder_conv3 = nn.Conv2d(128, 1, kernel_size=5, padding='same')
        
        #self.skip_connection = nn.Conv2d(128, 128, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.encoder_conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.encoder_conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.encoder_conv3(x))

       # e3= self.attention(e3)
        
        # Decoder
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.decoder_conv1(x))
        
        #skip_out = self.skip_connection(e2)
        
        #print('skip out shape is:', skip_out.shape)
       # print('d1 shape is:', d1.shape)
        #d1 = d1+skip_out
       # print('d1 out shape is :', d1.shape)
        
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.decoder_conv2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.sigmoid(self.decoder_conv3(x))
        
        return x

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice
from torch.utils.data import random_split

def main():
    root = '/home/ubuntu/fire/data/train'
    data = NpzDataset(root)

    # Split data into training and validation sets
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    # Create data loaders
    data_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    
    print(torch.cuda.is_available())

    # See comment below about epoch for loop
    #checkpoint = torch.load("checkpoints/checkpoint1_28.pth")

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    model = FCN()
   # model.load_state_dict(checkpoint["model_state_dict"])
    model.train()
    model.to(device)

    # Define a loss function and optimizer
    print("defining optimizer:")
    criterion = DiceLoss()
    lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7, verbose=True)
    patient_epochs = 10
    num_epochs = 200
    best_epoch_loss = np.inf

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_batches = len(data_loader)

        for batch_num, wrapped in enumerate(data_loader):
            t1 = time.time()
            inputs, targets = wrapped[0], wrapped[1]
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs.float())
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            t2 = time.time()
            if batch_num % 10 == 0:
                display.clear_output()
                display.display(f"Epoch Number: {epoch} ---- Batch: {batch_num} ---- Percentage done: {(batch_num / total_batches) * 100:.2f}% --- Loss: {loss.item():.4f} -- {t2-t1:.4f}")
        
        epoch_loss /= total_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(inputs.float())
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, targets.float())
                val_loss += loss.item()
                outputs.detach()
                inputs.detach()
                targets.detach()
                
        val_loss /= len(val_loader)
        # Adjust learning rate based on validation los
        scheduler.step(val_loss)
        print(f'learning rates: {[group["lr"] for group in optimizer.param_groups]}')

        if val_loss < best_epoch_loss:
            print(f"Model saved as checkpoint_Conv_mark3_{epoch}.pth for epoch {epoch}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
            }, f"checkpoints/checkpoint_Conv_mark3_{epoch}.pth")
            best_epoch_loss = val_loss
            patient_epochs = 0
        else:
            patient_epochs += 1
            
        if patient_epochs >= 10:
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

if __name__ == '__main__':
    main()
