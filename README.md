# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset


## DESIGN STEPS
### STEP 1:
Import the required PyTorch, Torchvision, NumPy, Matplotlib, and other necessary libraries for building the autoencoder.

### STEP 2: 
Load the MNIST dataset and convert the handwritten digit images into tensors for model processing.

### STEP 3: 
Add random Gaussian noise to the original images and keep the pixel values within the range 0 to 1.

### STEP 4:
Design the convolutional autoencoder with encoder layers for feature extraction and decoder layers for image reconstruction.

### STEP 5:
Train the model using MSE loss and the Adam optimizer to reconstruct clean images from noisy inputs.

### STEP 6: 
Test the trained model and visualize the original, noisy, and denoised images to check the performance.





## PROGRAM

### Name:HIBA NASREEN M

### Register Number:212224040117

```python
# Autoencoder for Image Denoising using PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transform: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])
# Load MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
# Add noise to images
def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)
# Define Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [1,28,28] -> [32,14,14]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [32,14,14] -> [64,7,7]
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [64,7,7] -> [32,14,14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # [32,14,14] -> [1,28,28]
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        # Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()               # Mean Squared Error for reconstruction
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Print model summary
print("Name: NIKILA D\nReg no: 212224230187")
summary(model, input_size=(1, 28, 28))
# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, images)   # Compare denoised output vs original clean image
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name:                   ")
    print("Register Number:                  ")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)


```

### OUTPUT

<img width="796" height="532" alt="image" src="https://github.com/user-attachments/assets/834f2806-31e4-4177-ac25-5c2226a0b0e1" />

<img width="406" height="125" alt="image" src="https://github.com/user-attachments/assets/101cd4a6-dc57-42d7-8d62-583770f37d3f" />

<img width="977" height="322" alt="image" src="https://github.com/user-attachments/assets/e98e574e-37c2-4f72-a9e2-b2e3be665332" />



## RESULT
The Convolutional Autoencoder was successfully trained to remove noise from MNIST images and reconstruct clear images with good accuracy.
