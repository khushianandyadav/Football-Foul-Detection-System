import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# -----------------------
# CONFIG
# -----------------------

BATCH_SIZE = 32
EPOCHS = 5
DATASET_PATH = "dataset"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# TRANSFORMS
# -----------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# -----------------------
# LOAD DATA
# -----------------------

dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print("Classes:", dataset.classes)

# -----------------------
# MODEL
# -----------------------

model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# -----------------------
# TRAINING LOOP
# -----------------------

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# -----------------------
# SAVE MODEL
# -----------------------

torch.save(model.state_dict(), "sport_classifier.pth")
print("\nModel saved as sport_classifier.pth")