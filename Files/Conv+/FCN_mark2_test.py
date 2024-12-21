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
# Define the device




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


class FCNWithSkipConnections(nn.Module):
    def __init__(self):
        super(FCNWithSkipConnections, self).__init__()
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
    

def main():
    test_dataset = NpzDataset('/Users/alexthomas/Desktop/EE_Data/test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dict = torch.load('/Users/alexthomas/Desktop/PyTorch/Clean data/checkpoint_Conv_mark2_24.pth', map_location=torch.device('cpu'))
    print(dict.keys())
    print(f'epoch: {dict["epoch"]}')
    print(f'loss {dict["loss"]}')
    model = FCNWithSkipConnections()
    model.load_state_dict(dict["model_state_dict"])
    model.eval()
    
    i = 250 # replace with the index of the desired batch
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
    ax1 = fig.add_subplot(3, 1, 1)  # Changed to 2x2 layout
    ax2 = fig.add_subplot(3, 1, 2)  # Changed to 2x2 layout
    ax3 = fig.add_subplot(3, 1, 3)  # New subplot for the first output  # New subplot for the second output
    
    
   
    ax1.imshow(inputs[0].cpu().detach().numpy(), cmap="Reds", alpha=0.5)
    ax2.imshow(target.cpu().detach().numpy(), cmap="Blues", alpha=0.5)
    ax3.imshow(result.cpu().detach().numpy(), cmap="Greens", alpha=0.5)
    
    plt.show()
        
if __name__ == '__main__':
    main()