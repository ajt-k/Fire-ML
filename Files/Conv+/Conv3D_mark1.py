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
        inputs = file['arr_0'][:11]
        outputs = file['arr_0'][11]

        input_tensor = torch.from_numpy(inputs).float()
        output_tensor = torch.from_numpy(outputs).float()
        
        seed = random.randint(0, 2 ** 32)
        
        input_tensor, output_tensor = self.random_crop_with_seed(input_tensor, output_tensor, (64, 64), seed)
        input_tensor, output_tensor = self.random_flip_with_seed(input_tensor, output_tensor, seed)

        fire = input_tensor[0:3]
        wind_u = input_tensor[3:8:2]
        wind_v = input_tensor[4:9:2]
        precip = input_tensor[9].unsqueeze(0)
        precip = precip.repeat(3,1,1)
        

        cover = input_tensor[10].unsqueeze(0)
        cover = cover.repeat(3,1,1)
        input_tensor = torch.stack([fire, wind_u, wind_v, precip, cover], dim=1)

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


class FCN_3d(nn.Module):
    def __init__(self):
        super(FCN_3d, self).__init__()
        
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(5, 32, kernel_size=(2, 3, 3), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(2, 3, 3), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        
        # Transition from 3D to 2D
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        
        # 2D Convolutional layers
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        
        # Final prediction layer
        self.final_conv = nn.Conv2d(128, 1, kernel_size=1)
        
    def forward(self, x):
        # Input shape: (batch_size, 3, 5, 64, 64)
        x = x.permute(0, 2, 1, 3, 4)  # Reshape to (batch_size, 5, 3, 64, 64)
        
        # 3D convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Transition from 3D to 2D
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.squeeze(2)  # Remove the temporal dimension, shape: (batch_size, 128, 64, 64)
        
        # 2D convolutions
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        

        x = torch.sigmoid(self.final_conv(x))
        
        return x


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()




#def loss_function(output, target):
#    bce_loss = nn.BCELoss(reduction= 'none')(output, target)
#    weights = 100 * target + (1-target)
#    weighted_loss = bce_loss*weights
#    return weighted_loss.sum()
#
from torch.utils.data import random_split

def train(model, data_loader, criterion, optimizer, device):
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
            display.display(f"Batch: {batch_num} ---- Percentage done: {(batch_num / total_batches) * 100:.2f}% --- Loss: {loss.item():.4f} -- {t2-t1:.4f}")

        outputs.detach()
        inputs.detach()
        targets.detach()
    epoch_loss /= total_batches
    return epoch_loss

def validate(model, val_loader, criterion, device):
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
    return val_loss


from torch.optim.lr_scheduler import CyclicLR


def main():
    root = '/home/ubuntu/fire/data/train_3'
    data = NpzDataset(root)

    # Split data into training and validation sets
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    # Create data loaders
    data_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    print(torch.cuda.is_available())

    device = "cuda" 
    
    checkpoint = torch.load('checkpoints/checkpoint_Conv3d_mark1_30.pth')
    
    model = FCN_3d()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    model.to(device)

    # Define a loss function and optimizer
    print("defining optimizer:")
    criterion = FocalLoss()
    
    lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    for param_group in optimizer.param_groups:
        param_group['lr'] == lr

     ##L2 Regulaization
    
    
    # Learning rate scheduler
    #scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, step_size_up=10, mode='triangular')
    
    num_epochs = 200
    best_epoch_loss = checkpoint['loss']#np.inf
    starting_epoch = checkpoint['epoch']

    for epoch in range(num_epochs):
        epoch_loss = train(model, data_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        # Adjust learning rate based on validation loss
       # scheduler.step(val_loss)
        print(f'learning rates: {[group["lr"] for group in optimizer.param_groups]}')

        if val_loss < best_epoch_loss:
            print(f"Model saved as checkpoint_Conv3d_mark1_{epoch+starting_epoch}.pth for epoch {epoch+starting_epoch}")
            torch.save({
                "epoch": epoch+starting_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
            }, f"checkpoints/checkpoint_Conv3d_mark1_{epoch+starting_epoch}.pth")
            best_epoch_loss = val_loss
            patient_epochs = 0
        else:
            patient_epochs += 1
            
        if patient_epochs >= 15:
            print(f"Early stopping at epoch {epoch+starting_epoch}")
            break

        print(f"Epoch {epoch+starting_epoch}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

if __name__ == '__main__':
    main()

