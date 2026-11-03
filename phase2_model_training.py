
# ==========================================
# PHASE 2: BUILDING & TRAINING CNN MODEL
# ==========================================

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# 1. CHECK DEVICE
# Use GPU if available, otherwise CPU

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# 2. LOAD PROCESSED DATA

data = np.load("mnist_processed.npz")

X_train = torch.tensor(data["X_train"], dtype=torch.float32)
y_train = torch.tensor(data["y_train"], dtype=torch.long)

X_val = torch.tensor(data["X_val"], dtype=torch.float32)
y_val = torch.tensor(data["y_val"], dtype=torch.long)

# 3. CREATE DATALOADERS

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

# 4. BUILD CNN MODEL

class DigitCNN(nn.Module):

    def __init__(self):
        super(DigitCNN, self).__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, padding=1
        )

        # Second convolution layer
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1
        )

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        # Convolution + ReLU + Pooling
        x = self.pool(self.relu(self.conv1(x)))

        # Second convolution + ReLU + Pooling
        x = self.pool(self.relu(self.conv2(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))

        # Final output (10 digits)
        x = self.fc2(x)

        return x


# 5. CREATE MODEL

model = DigitCNN().to(device)

print("\nCNN Model Created Successfully!")
print(model)

# 6. DEFINE LOSS FUNCTION AND OPTIMIZER

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# 7. TRAIN THE MODEL

epochs = 5

train_losses = []
val_accuracies = []

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    train_losses.append(avg_loss)

    # 8. VALIDATE THE MODEL

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            total += labels.size(0)

            correct += (predictions == labels).sum().item()

    accuracy = 100 * correct / total

    val_accuracies.append(accuracy)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {avg_loss:.4f} "
        f"Validation Accuracy: {accuracy:.2f}%"
    )

# 9. SAVE TRAINED MODEL

torch.save(
    model.state_dict(),
    "digit_cnn.pth"
)

print("\nModel Saved Successfully!")
print("File Name: digit_cnn.pth")

# 10. DISPLAY TRAINING RESULTS

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, marker="o")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, marker="o")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.tight_layout()
plt.show()
