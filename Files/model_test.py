import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom

import pdb

import torch.nn as nn
import torch.nn.functional as F



class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        #could change other factors about this such as dialation and the stride
        self.encoder_conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.decoder_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.encoder_conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.encoder_conv2(x))
        x = F.max_pool2d(x, 2)

        # Decoder
            #This doubles the scale through interpolation
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        #non linearity
        x = F.relu(self.decoder_conv1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        #squishes between 0 and 1
        x = torch.sigmoid(self.decoder_conv2(x))

        return x


def main():
    # Assuming you have a model instance and input data
    model = FCN()  # Replace with your model's initialization
    list  = torch.load('/Users/alexthomas/Desktop/PyTorch/checkpoint.pth')
    print(type(list))
    print(list.keys())
    state_dict = list['model_state_dict']
    
    print('\n\ntype of state_dict:', type(state_dict))
    print('\n\nstate_dict keys:', state_dict.keys())
    #print(list)

    checkpoint = torch.load('/Users/alexthomas/Desktop/PyTorch/checkpoint.pth', map_location=torch.device('cpu'))

# If you want to use the model for inference, you should also call model.eval()
    
    #model.eval()
    
    
    print(type(checkpoint))
    pdb.set_trace()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('3')

if __name__ == "__main__":
    main()

