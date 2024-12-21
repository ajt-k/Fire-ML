import torch
from torch.utils.data import DataLoader
from MultiConvLSTM.py import MultiKernelConvLSTM  # Assuming this is the correct import for your model

def test_single_instance(data):
    # Assuming 'data' is your dataset object already loaded

    # Create a DataLoader for one instance
    data_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiKernelConvLSTM(input_channels=4, hidden_channels=[128, 64, 64], kernel_sizes=[(3, 3), (5, 5), (5, 5)])
    model.eval()  # Set the model to evaluation mode
    model.to(device)

    # Fetch one instance
    for instance in data_loader:
        instance = [i.to(device) for i in instance]  # Move instance to the correct device
        output = model(instance)
        print("Output shape:", output.shape)
        break  # Only process one instance

if __name__ == '__main__':
    # Assuming 'data' is defined elsewhere in your script
    test_single_instance(data)