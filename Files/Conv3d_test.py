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
from torch.optim.lr_scheduler import CyclicLR


def main():
    root = '/Users/alexthomas/Desktop/PyTorch/Clean data'
    data = NpzDataset(root)
    test_loader = DataLoader(data, batch_size=1, shuffle=False)

    dict = torch.load('/Users/alexthomas/Downloads/checkpoint_Conv3d_mark2_14.pth', map_location=torch.device('cpu'))
    print(dict.keys())
    print(f'epoch: {dict["epoch"]}')
    print(f'loss {dict["loss"]}')
    
    model = FCN_3d()
    model.load_state_dict(dict["model_state_dict"])
    model.eval()
    
    i = 0 # replace with the index of the desired batch
    for batch_index, (inputs, labels) in enumerate(test_loader):
        if batch_index == i:
         break
    
    
    target = labels.squeeze()  # Similarly, remove batch dimension for output if batch_size=1

    result = model(inputs)
    result = result.squeeze(0)
    result = result.squeeze(0)
    print(f'result shape: {result.shape}')
    
    inputs = inputs.squeeze(0)
    print(f'inputs shape {inputs.shape}')
    
    print(f'target shape{target.shape}')
    
    fig = plt.figure(figsize=(10, 4))  # Adjust figure size for better visualization
    
    # Displaying input images
    ax1 = fig.add_subplot(2,3 , 1)  # Changed to 2x2 layout
    ax1.set_title('Input')
    ax2 = fig.add_subplot(2,3 , 2)  # Changed to 2x2 layout
    ax2.set_title('Target')
    ax3 = fig.add_subplot(2,3 , 3)  # New subplot for the first output  # New subplot for the second output
    ax3.set_title('Output')
    ax4 = fig.add_subplot(2,3 , 4)  # New subplot for the second output
    ax4.set_title('wind_u')
    
    ax5 = fig.add_subplot(2,3 , 5)  # New subplot for the second output
    ax5.set_title('wind_v')
    ax6 = fig.add_subplot(2,3 , 6)  # New subplot for the second output
    ax6.set_title('cover')
    
    
   
    ax1.imshow(inputs[2][0].cpu().detach().numpy(), cmap="Reds", alpha=0.5)
    ax1.imshow(inputs[1][0].cpu().detach().numpy(), cmap="Blues", alpha=0.5)
    ax1.imshow(inputs[0][0].cpu().detach().numpy(), cmap="Greens", alpha=0.1)
    ax2.imshow(target.cpu().detach().numpy(), cmap="Blues", alpha=0.5)
    ax3.imshow(result.cpu().detach().numpy(), cmap="Reds")
    
    ax4.imshow(inputs[2][1].cpu().detach().numpy(), cmap="Reds", alpha=0.3)
    ax4.imshow(inputs[1][1].cpu().detach().numpy(), cmap="Blues", alpha=0.3)
    ax4.imshow(inputs[0][1].cpu().detach().numpy(), cmap="Greens", alpha=0.3)
    
    ax5.imshow(inputs[2][2].cpu().detach().numpy(), cmap="Reds", alpha=0.3)
    ax5.imshow(inputs[1][2].cpu().detach().numpy(), cmap="Blues", alpha=0.3)
    ax5.imshow(inputs[0][2].cpu().detach().numpy(), cmap="Greens", alpha=0.3)
    
    ax6.imshow(inputs[2][4].cpu().detach().numpy(), cmap="Greens")
    print(np.min(result.cpu().detach().numpy()))
    print(np.max(result.cpu().detach().numpy()))
    
    fig2 = plt.figure(figsize=(10, 4))
    ax21 = fig2.add_subplot(1,2,1)
    ax22 = fig2.add_subplot(1,2,2)
    ax21.imshow(result.cpu().detach().numpy()>0.8, cmap="Reds", alpha = 0.3)
    ax21.imshow(result.cpu().detach().numpy()>0.9, cmap="Reds", alpha = 0.3)
    ax21.imshow(result.cpu().detach().numpy()>0.94, cmap="Reds", alpha = 0.3)
    ax22.imshow(result.cpu().detach().numpy()>0.96, cmap="Reds", alpha = 0.3)
    ax22.imshow(result.cpu().detach().numpy()>0.98, cmap="Reds", alpha = 0.3)
    ax22.imshow(result.cpu().detach().numpy()>0.99, cmap="Reds", alpha = 0.3)

    plt.show()
        
if __name__ == '__main__':
    main()

