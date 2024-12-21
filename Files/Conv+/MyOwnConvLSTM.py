

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


class ConvLSTM(nn.Module):
    def __init__(self, num_layers, seq_len, input_channels, kernels, hidden_channels):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        #3
        self.seq_len = seq_len
        #[3,5,7]
        
        self.kernels = kernels
        
        #[5,128, 256]
        self.initial_channels = input_channels
        
        #[128, 256, 512]
        self.hidden_channels = hidden_channels
        
        lst = [nn.ModuleList([nn.Conv2d(in_channels=hidden_channels[j]+input_channels[j], 
                         out_channels = 4*hidden_channels[j], 
                         kernel_size=kernels[i],
                         padding=kernels[i]//2) for i in range(len(input_channels))]) for j in range(seq_len) ]
        
        self.convlstm_cells = nn.ModuleList(lst)
        self.final_layer = nn.Conv2d(256, 1, kernel_size=1, padding='same')
        
        
     
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        total_states = []
        
        for i in range(0, len(self.hidden_channels)):
            #This should get the hidden channels for each layer, and then the first device for each row of cells
            h = torch.zeros(batch_size, self.hidden_channels[i], height, width, device=self.convlstm_cells[i][0].weight.device)
            c = torch.zeros(batch_size, self.hidden_channels[i], height, width, device=self.convlstm_cells[i][0].weight.device)
            total_states.append([h,c])
        return total_states
    
    def forward(self, input_tensor):
        #print('starting forward call:')
        batch_size, seq_len, _, height, width = input_tensor.size()
        
        #print(f'input tensor size: {input_tensor.size()}')

        hidden_channels =  self.init_hidden(batch_size, (height, width))
        #print(f'len of hidden channels should be 3: {len(hidden_channels)}')
        
        #this will feed old hidden states into the second and third layers
        layer_states = []
        last_layer_state = []
        
        #iterates over number of kernels
        for row in range(len(hidden_channels)):
            
            
            last_layer_state = []
            #gets hidden_channels at 
            h_cur,c_cur = hidden_channels[row]
            
            if row == 0: 
                x = input_tensor
            else:
                x = torch.stack([layer_states[row-1][counter][0] for counter in range(3)], dim=1)
         
            for col, cell in enumerate(self.convlstm_cells[row]):
                #print(f'layer {row}, cell {col}')
                
                #print(f'conv weight size {cell.weight.size()}')
                
                #input data for layer -- first pass will be 5, 128, 256
                tensor = x[:,col,:,:,:]
                #print('tensor size', tensor.size())
                
                combined = torch.cat([tensor, h_cur], dim=1)  # concatenate along channel axiss
                #print(f'combined shape {combined.shape}')
                combined_conv = cell(combined)
                #print(f'combined_conv {combined_conv.shape}')
                #print(f'hidden channels {self.hidden_channels[row]}')
                cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels[row], dim=1)
                #print(f'shape cc_i {cc_i.shape}')
                #print(f'shape cc_f {cc_f.shape}')


                #update according to LSTM equations
                i = torch.sigmoid(cc_i)
                f = torch.sigmoid(cc_f)
                o = torch.sigmoid(cc_o)
                g = torch.tanh(cc_g)
                c_next = f * c_cur + i * g
                h_next = o * torch.tanh(c_next)
                
                last_layer_state.append([h_next, c_next])
                
                #print(f'c_next shape: {c_next.shape}')
                #print(f'h_next shape: {h_next.shape}')
                
                #update current total state to reflect updated valyes
                h_cur = h_next
                c_cur = c_next
                
            layer_states.append(last_layer_state)

                
        #should return (h,c) of the final list
        output = torch.sigmoid(self.final_layer(layer_states[2][2][0]))
        return output
        
    
from torch.utils.data import random_split

def main():
    root = '/home/ubuntu/fire/data/train_3'
    data = NpzDataset(root)

    # Split data into training and validation sets
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    # Create data loaders
    data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    model = ConvLSTM(num_layers=3, seq_len=3, input_channels=[5, 128, 256], hidden_channels=[128, 256, 256], kernels=[3, 5, 5])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

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
            if batch_num % 5 == 0:
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
        val_loss /= len(val_loader)

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

        if val_loss < best_epoch_loss:
            print(f"Model saved as checkpoint_ConvLSTM_LR_{epoch}.pth for epoch {epoch}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
            }, f"checkpoints/checkpoint_ConvLSTM_LR_{epoch}.pth")
            best_epoch_loss = val_loss

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

if __name__ == '__main__':
    main()

