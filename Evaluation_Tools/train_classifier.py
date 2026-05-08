import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
import torchvision.transforms.functional as TF
import copy

# Configuration
DATA_DIR = r"MSTAR\PERSONAL_MSTAR\45_DEG"  # Dataset is inside the project root, not parent
SAVE_PATH = "checkpoints/classifier_resnet18_best.pth"
IMG_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 30  # Sufficient for MSTAR
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PadToSquare:
    def __call__(self, img):
        # img is a PIL Image
        w, h = img.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return TF.pad(img, padding, 0, 'constant')


def main():
    print(f"Using device: {DEVICE}")
    os.makedirs("checkpoints", exist_ok=True)

    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        PadToSquare(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset not found at {DATA_DIR}")
        return

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Classes found: {class_names}")

    # Split Train/Val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }
    dataset_sizes = {'train': train_size, 'val': val_size}

    # Model Setup (ResNet18)
    # We modify the first conv layer to accept 1 channel instead of 3
    model = models.resnet18(pretrained=False)  # Train from scratch is usually better for SAR than ImageNet
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            running_loss = 0.0
            running_corrects = 0
            
            # Use enumerate to track batch index
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'train' and (i + 1) % 10 == 0:
                     print(f"  [Batch {i + 1}/{len(dataloaders[phase])}] Loss: {loss.item():.4f}")

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, SAVE_PATH)
                print(f"New best model saved! Acc: {best_acc:.4f}")

    print(f"Training complete. Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
