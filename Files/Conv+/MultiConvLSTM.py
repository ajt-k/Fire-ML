

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
    
    
    
    
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels, out_channels= 4*(self.hidden_channels), kernel_size = self.kernel_size, padding=self.kernel_size[0] // 2)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        print(h_cur.shape)
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axiss
        print(f'combined shape {combined.shape}')
        combined_conv = self.conv(combined)
        print(f'combined_conv {combined_conv.shape}')
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        print(f'shape cc_i {cc_i.shape}')
        print(f'shape cc_f {cc_f.shape}')
        
        
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        print(f'c_next shape: {c_next.shape}')
        print(f'h_next shape: {h_next.shape}')
        return h_next, c_next

  
    
class MultiKernelConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_sizes):
        super(MultiKernelConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_sizes = kernel_sizes
        self.convlstm_cells = nn.ModuleList([ConvLSTMCell(self.input_channels, self.hidden_channels[i], self.kernel_sizes[i]) for i in range(len(self.kernel_sizes))])

    def forward(self, input_tensor, hidden_state=None):
        print('inputs tensor size', input_tensor.size())
        batch_size, seq_len, _, height, width = input_tensor.size()
        
        layers = 3
        if hidden_state is None:
            init_states = self.init_hidden(batch_size, (height, width))
        
        layer_states = []
        hidden_channels = [128, 256, 512]
        
        for j in range(layers):
            
            #h0, c0
            h, c = init_states[j]
            #this is what will be passed to next layer
            
            hidden_states = []
            if j ==0:
                for i, cell in enumerate(self.convlstm_cells):
                    #for the first pass, use the real data
                    x = input_tensor[:,i,:,:,:]
                    h, c = cell(x, (h, c))
                    hidden_states.append([h,c])
            else:
                for i, cell in enumerate(self.convlstm_cells):
                    x = layer_states[i]
                    h, c = cell(x, (h, c))
            #This allows me to have the hidden states separate to avoid editing the same list you are taking from
            layer_states.append(hidden_states)
         

       # output_inner = []
        #for i in range(seq_len):
        #    x = input_tensor[:, i, :, :, :]
        #    for j, cell in enumerate(self.convlstm_cells):
        #        if i == 0:
        #            h, c = hidden_states[j]
        #        else:
        #            h, c = output[j]
        #        output, (h, c) = cell(x, (h, c))
        #        x = output
            #output_inner.append(x)
        for i, cell in enumerate(self.convlstm_cells):
            x = input_tensor[:,i,:,:,:]
            
            h, c = cell(x, (h, c))
                
        return x, (h, c)

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        total_states = []
        for i in range(0, len(self.hidden_channels)):
            h = torch.zeros(batch_size, self.hidden_channels[i], height, width, device=self.convlstm_cells[i].conv.weight.device)
            c = torch.zeros(batch_size, self.hidden_channels[i], height, width, device=self.convlstm_cells[i].conv.weight.device)
            total_states.append([h,c])
        return total_states


    
    
def visualize_transformed_images(dataloader):
    dataiter = iter(dataloader)
    img, output = next(dataiter)  # Assuming the dataloader now also returns an output tensor
    img = img.squeeze()  # Remove batch dimension if batch_size=1
    output = output.squeeze()  # Similarly, remove batch dimension for output if batch_size=1
    
    fig = plt.figure(figsize=(10, 4))  # Adjust figure size for better visualization
    
    # Displaying input images
    ax1 = fig.add_subplot(2, 2, 1)  # Changed to 2x2 layout
    ax2 = fig.add_subplot(2, 2, 2)  # Changed to 2x2 layout
    ax1.imshow(img[0], cmap="Reds")
    ax2.imshow(img[1], cmap="Reds")
    
    # Displaying corresponding outputs
    ax3 = fig.add_subplot(2, 2, 3)  # New subplot for the first output
    ax4 = fig.add_subplot(2, 2, 4)  # New subplot for the second output
    ax3.imshow(output, cmap="Reds")
    #ax4.imshow(output, cmap="Reds")
    
    plt.axis('off')  # Hide axes for better visualization
    plt.show()
    
def main():
    root = '/Users/alexthomas/Desktop/PyTorch/Clean data'
    data = NpzDataset(root)

    print(f"Number of items: {len(data)}")

    # Create data loaders
    print('loading model and Dataloader: ')
    data_loader = DataLoader(data, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)
    
    

    
    model = MultiKernelConvLSTM(input_channels=5, hidden_channels=[128, 256, 512], kernel_sizes=[(3, 3), (5, 5), (5, 5)])
    #output, (h, c) = multi_kernel_convlstm(input_tensor)
    #model.load_state_dict(checkpoint["model_state_dict"])
    model.train()
  
    
    data = next(iter(data_loader))
    input_tensor, output_tensor = data
    criterion = nn.MSELoss()  # Example loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (input_data, target) in enumerate(data_loader):
            optimizer.zero_grad()

            # Forward pass
            h_state, c_state = model(input_data)
            
            print('output', h_state.shape)
            print('target,', h_state.shape)
            # Calculate loss
            loss = criterion(h_state, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")  

    
    


if __name__ == '__main__':
    main()
    
