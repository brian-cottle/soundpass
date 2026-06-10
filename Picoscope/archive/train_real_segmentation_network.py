# %%

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import torch.optim as optim
from glob import glob
from torchinfo import summary
from tqdm import tqdm
import os

# %% # Dataset class
class SegmentationDataset(Dataset):
    def __init__(self,
                 file_paths=['/media/briancottle/Samsung_T5/SoundPass_datasets/cadaver_annotated_datasets',
                             ],
                 split='Train',
                 percentage=1.0,
                 baseline=False,
                 normalize=True,
                 downsample_rate=1):
        assert 0 <= percentage <= 1, "Percentage must be between 0 and 1"
        self.split = split
        self.baseline = baseline
        self.normalize = normalize
        self.downsample_rate = downsample_rate

        all_file_names = []

        for file_path in file_paths:
            pkl_files = glob(f"{file_path}/**/*.pkl",recursive=True)
            all_file_names.extend(pkl_files)

        split_idx = int(0.95 * len(all_file_names))
        val_idx = int(0.99 * len(all_file_names))

        if self.split == 'Train':
            all_file_names = all_file_names[:split_idx]
        elif self.split == 'Val':
            all_file_names = all_file_names[split_idx:val_idx]
        elif self.split == 'Test':
            all_file_names = all_file_names[val_idx:]

        # Apply percentage filter
        num_files_to_keep = int(len(all_file_names) * percentage)
        self.names = all_file_names[:num_files_to_keep]
        self.baseline_array = None



    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        with open(self.names[idx], 'rb') as f:
            sample_data = pickle.load(f)
        data = np.array(sample_data['data'])
        gt = np.array(sample_data['ground_truth'])

        if self.baseline:
            data = data - self.baseline_array

        if self.normalize:
            data = (data - np.mean(data)) / np.std(data)

        if self.downsample_rate > 1:
            data = data[::self.downsample_rate]
            gt = gt[::self.downsample_rate]


        return data, gt

# Define a simple 1D segmentation network
class SegmentationNetwork(nn.Module):
    def __init__(self, input_channels, num_classes,kernel_size=25,hidden_channels=64):
        super(SegmentationNetwork, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_channels, num_classes, kernel_size=kernel_size, padding='same')
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return self.softmax(x)
    
# %%

input_channels = 1
num_classes = 2
sequence_length = 30000
batch_size = 16
num_epochs = 100
learning_rate = 0.001
percentage = 1
downsample_rate = 30
kernel_size = 45
hidden_channels = 85
# Dataset and DataLoader
train_dataset = SegmentationDataset(percentage=percentage, split='Train', baseline=False, downsample_rate=downsample_rate)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Validation dataset and DataLoader
val_dataset = SegmentationDataset(split='Val',percentage=percentage, baseline=False, downsample_rate=downsample_rate)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True) 

# Test dataset and DataLoader   
test_dataset = SegmentationDataset(split='Test',percentage=percentage, baseline=False, downsample_rate=downsample_rate)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print(f'Number of training samples: {len(train_dataset)}')
print(f'Number of validation samples: {len(val_dataset)}')
print(f'Number of test samples: {len(test_dataset)}')

# %%
all_sample_ratios = []
for idx,sample in enumerate(train_dataset):
    data, gt = sample
    positive_count = np.sum(gt == 1)
    negative_count = np.sum(gt == 0)
    # if positive_count > 0:
    #     plt.figure(figsize=(10, 2))
    #     plt.plot(gt, label="Ground Truth")
    #     plt.title(f"Sample {idx} - Positive Classification")
    #     plt.xlabel("Time")
    #     plt.ylabel("Class")
    #     plt.legend()
    #     plt.show()
    #     plt.pause(5)
    ratio = positive_count / negative_count if negative_count > 0 else float('inf')
    all_sample_ratios.append(ratio)

print(f'mean ratios: {np.mean(all_sample_ratios)}')

ratio = np.mean(all_sample_ratios)
# ratio = 0.2

# %%
weight_tensor = torch.tensor([ratio*100, (1-ratio)*100]).float()
# Model, loss, and optimizer

model = SegmentationNetwork(input_channels, num_classes,hidden_channels=hidden_channels, kernel_size=kernel_size)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)  # Adjust weights for class imbalance
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
print(summary(model, input_size=(batch_size, input_channels, sequence_length)))

# Create a directory to save model weights

save_dir = '/home/briancottle/Code/SoundPass/cadaver_model_weights/v0_2'
os.makedirs(save_dir, exist_ok=True)
# Save hyperparameters to a text file

hyperparameters = {
    "input_channels": input_channels,
    "num_classes": num_classes,
    "sequence_length": sequence_length,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "percentage": percentage,
    'downsample_rate': downsample_rate,
    "kernel_size": kernel_size,
    "hidden_channels": hidden_channels,
    "ratio": ratio,
}

hyperparameters_file = os.path.join(save_dir, "log_file.txt")
if os.path.exists(hyperparameters_file):
    with open(hyperparameters_file, "a") as f:
        f.write(str(hyperparameters) + "\n")
else:
    with open(hyperparameters_file, "w") as f:
        f.write(str(hyperparameters) + "\n")

print(f"Hyperparameters saved to {hyperparameters_file}")

# %%
# Training loop
for epoch in tqdm(range(num_epochs)):
    model.train()
    for batch_data, batch_labels in dataloader:
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # Reshape data to match the input shape of the model
            batch_labels = batch_labels.long()  # Ensure labels are of type lonog_fileg

            # Forward pass
            outputs = model(batch_data.unsqueeze(1).float())
            loss = criterion(outputs, batch_labels)

            # Backward pass and optimization
            
            loss.backward()
            optimizer.step()

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_data, val_labels in val_dataloader:
            val_labels = val_labels.long()  # Ensure labels are of type long

            # Forward pass
            val_outputs = model(val_data.unsqueeze(1).float())
            val_loss += criterion(val_outputs, val_labels).item()

    val_loss /= len(val_dataloader)  # Average validation loss
    scheduler.step()  # Update learning rate
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
    
    # Log the epoch, training loss, and validation loss to the log file
    with open(hyperparameters_file, "a") as f:
        f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}\n")
    # Save the model weights
    torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))


# %%

# Load model weights from a given file name
model_weights_file = os.path.join(save_dir, "model_epoch_35.pth")  # Replace with the desired file name
if os.path.exists(model_weights_file):
    model.load_state_dict(torch.load(model_weights_file))
    model.eval()
    print(f"Model weights loaded from {model_weights_file}")
else:
    print(f"Model weights file {model_weights_file} not found.")


# %%

train_iter = iter(dataloader)
train_data, train_labels = next(train_iter)
train_labels = train_labels.long()  # Ensure labels are of type long

for sample_idx in range(len(train_data)):
    # Select a sample from the batch
    sample_data = train_data[sample_idx].unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
    sample_label = train_labels[sample_idx].unsqueeze(0)  # Add batch dimension

    # Run inference
    model.eval()
    with torch.no_grad():
        sample_output = model(sample_data)
        sample_loss = criterion(sample_output, sample_label)

    # Convert tensors to numpy arrays for plotting
    sample_data_np = sample_data.squeeze().numpy()
    sample_output_np = sample_output.squeeze().numpy()
    sample_label_np = sample_label.squeeze().numpy()

    # Plot the input signal, output logits, and ground truth
    plt.figure(figsize=(15, 5))

    # Input signal
    plt.subplot(3, 1, 1)
    plt.plot(sample_data_np)
    plt.title("Input Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    # Output logits
    plt.subplot(3, 1, 2)
    plt.plot(sample_output_np[0], label="Class 0")
    plt.plot(sample_output_np[1], label="Class 1")
    plt.title("Output Logits")
    plt.xlabel("Time")
    plt.ylabel("Logit Value")
    plt.legend()

    # Ground truth
    plt.subplot(3, 1, 3)
    plt.plot(sample_label_np)
    plt.title("Ground Truth")
    plt.xlabel("Time")
    plt.ylabel("Class")

    plt.tight_layout()
    plt.show()

    print(f"Loss for the selected sample: {sample_loss.item():.4f}")


# %%

test_iter = iter(test_dataloader)
test_data, test_labels = next(test_iter)
test_labels = test_labels.long()  # Ensure labels are of type long


for sample_idx in range(len(test_data)):
    # Select a sample from the batch
    sample_data = test_data[sample_idx].unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
    sample_label = test_labels[sample_idx].unsqueeze(0)  # Add batch dimension

    # Run inference
    model.eval()
    with torch.no_grad():
        sample_output = model(sample_data)
        sample_loss = criterion(sample_output, sample_label)

    # Convert tensors to numpy arrays for plotting
    sample_data_np = sample_data.squeeze().numpy()
    sample_output_np = sample_output.squeeze().numpy()
    sample_label_np = sample_label.squeeze().numpy()

    # Plot the input signal, output logits, and ground truth
    plt.figure(figsize=(15, 5))

    # Input signal
    plt.subplot(3, 1, 1)
    plt.plot(sample_data_np)
    plt.title("Input Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    # Output logits
    plt.subplot(3, 1, 2)
    plt.plot(sample_output_np[0], label="Class 0")
    plt.plot(sample_output_np[1], label="Class 1")
    plt.title("Output Logits")
    plt.xlabel("Time")
    plt.ylabel("Logit Value")
    plt.legend()

    # Ground truth
    plt.subplot(3, 1, 3)
    plt.plot(sample_label_np)
    plt.title("Ground Truth")
    plt.xlabel("Time")
    plt.ylabel("Class")

    plt.tight_layout()
    plt.show()

    print(f"Loss for the selected sample: {sample_loss.item():.4f}")

# %%
