
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset, TensorDataset
from torchvision import transforms, datasets, models
import torchvision.transforms.functional as TF
import copy
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

# ============================================================
# User Configuration - EDIT PATHS HERE
# ============================================================
# 1. Dataset Path (Real MSTAR Data)
REAL_DATA_DIR = r"..\MSTAR\PERSONAL_MSTAR\15_DEG" 

# 2. Generator Checkpoint Path (For Augmentation Experiment)
# Change this to point to the generator you want to test (e.g. DiffAug, Physics)
GENERATOR_CHECKPOINT = r"..\Test_Results\Test_ACGAN_WGAN_GP_DiffAug_Results\generator_final.pth"

# 3. Output Directory for Evaluation Results
EVAL_RESULT_DIR = r"..\Evaluation_Results"

# 4. Model Architecture Type (for loading generator correctly)
# Options: 'ACGAN_WGAN_GP', 'ACGAN_WGAN_GP_DiffAug', 'ACGAN_WGAN_GP_Physics'
GENERATOR_TYPE = 'ACGAN_WGAN_GP_DiffAug' 

# 5. Augmentation Settings
AUGMENT_RATIO = 5.0       # How many fake images to add relative to real training set (e.g. 1.0 = equal amount)
REAL_TRAIN_RATIO = 0.1    # Use only 10% of real training data to simulate data scarcity
BATCH_SIZE = 64
EPOCHS = 30               # Training epochs for the classifier
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Helper Functions & Classes
# ============================================================

def get_generator_class(gen_type):
    """
    Dynamically loads the correct Generator class based on type string using importlib.
    This avoids sys.path issues and "Unresolved reference" errors at runtime.
    """
    # Calculate absolute path to the cDCGAN directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cdcgan_dir = os.path.join(current_dir, "..", "cDCGAN")
    
    # Map gen_type to filename
    if gen_type == 'ACGAN_WGAN_GP':
        filename = "ACGAN_WGAN_GP.py"
    elif gen_type == 'ACGAN_WGAN_GP_DiffAug':
        filename = "ACGAN_WGAN_GP_DiffAug.py"
    elif gen_type == 'ACGAN_WGAN_GP_Physics':
        filename = "ACGAN_WGAN_GP_Physics.py"
    else:
        raise ValueError(f"Unknown generator type: {gen_type}")

    file_path = os.path.join(cdcgan_dir, filename)
    
    if not os.path.exists(file_path):
         raise FileNotFoundError(f"Generator file not found at: {file_path}")

    # Dynamic Import
    # Add cDCGAN to sys.path temporarily so that internal imports (like DiffAugment_pytorch) work
    if cdcgan_dir not in sys.path:
        sys.path.insert(0, cdcgan_dir)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location(gen_type, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[gen_type] = module
    spec.loader.exec_module(module)
    
    return module.Generator, module.LATENT_DIM, module.IMG_SIZE

class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return TF.pad(img, padding, 0, 'constant')

def load_real_data():
    """Loads and splits real data into Train (80%) and Test (20%)"""
    if not os.path.exists(REAL_DATA_DIR):
        print(f"Error: Dataset not found at {REAL_DATA_DIR}")
        sys.exit(1)
        
    print(f"Loading Real Data from: {REAL_DATA_DIR}")
    
    # Preprocessing
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        PadToSquare(),
        transforms.Resize((64, 64)), # Hardcoded size usually fine for MSTAR
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    full_dataset = datasets.ImageFolder(REAL_DATA_DIR, transform=data_transforms)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    
    # Split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    
    # Fixed seed for reproducibility of split
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42) 
    )

    # Secondary split to simulate data scarcity if REAL_TRAIN_RATIO < 1.0
    if REAL_TRAIN_RATIO < 1.0:
        small_train_size = int(len(train_dataset) * REAL_TRAIN_RATIO)
        # Ensure at least one sample per class roughly, but we just split
        small_train_size = max(small_train_size, num_classes * 5) 
        discard_size = len(train_dataset) - small_train_size
        train_dataset, _ = random_split(
            train_dataset,
            [small_train_size, discard_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    print(f"Dataset Split (After Scarce Reduction): Train={len(train_dataset)}, Test={len(test_dataset)}")
    return train_dataset, test_dataset, num_classes, class_names

def generate_fake_data(num_samples_per_class, num_classes, gen_checkpoint, gen_type, device):
    """
    Loads the generator and creates a synthetic dataset.
    """
    print(f"\n[Augmentation] Generating {num_samples_per_class} fake samples per class...")
    print(f"Loading Generator from: {gen_checkpoint}")
    
    if not os.path.exists(gen_checkpoint):
        print(f"Error: Generator checkpoint not found!")
        sys.exit(1)
        
    GeneratorClass, latent_dim, img_size = get_generator_class(gen_type)
    
    netG = GeneratorClass(num_classes).to(device)
    netG.load_state_dict(torch.load(gen_checkpoint, map_location=device))
    netG.eval()
    
    fake_images_list = []
    fake_labels_list = []
    
    batch_size = 50 # Generating in chunks
    
    with torch.no_grad():
        for c in range(num_classes):
            count = 0
            while count < num_samples_per_class:
                current_batch = min(batch_size, num_samples_per_class - count)
                
                # Input noise
                z = torch.randn(current_batch, latent_dim).to(device)
                labels = torch.full((current_batch,), c, dtype=torch.long).to(device)
                
                # Generate
                # Handle potentially different forward signatures
                try:
                    # Try standard forward
                    imgs = netG(z, labels)
                except TypeError:
                    # Physics model might need extra args, but evaluation usually works with default
                    # If using annealed physics, passing default epoch usually generates full noise usage
                    imgs = netG(z, labels)
                
                fake_images_list.append(imgs.cpu())
                fake_labels_list.append(labels.cpu())
                
                count += current_batch
                
    # Concatenate all
    all_fake_imgs = torch.cat(fake_images_list, dim=0)
    all_fake_labels = torch.cat(fake_labels_list, dim=0)
    
    print(f"Generated Total: {len(all_fake_imgs)} images.")
    
    # Create a custom Dataset that returns data in the same format as ImageFolder
    # ImageFolder returns (tensor, int), not (tensor, tensor)
    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
            
        def __len__(self):
            return len(self.images)
            
        def __getitem__(self, idx):
            # Return (image_tensor, label_as_int) to match ImageFolder format
            return self.images[idx], int(self.labels[idx])
    
    return FakeDataset(all_fake_imgs, all_fake_labels)

def train_classifier(train_dataset, test_dataset, num_classes, experiment_name, writer=None, writer_step_offset=0):
    """
    Trains a ResNet18 classifier from scratch and evaluates it.
    Returns best accuracy.
    """
    print(f"\n{'='*40}")
    print(f"Training Classifier: {experiment_name}")
    print(f"{'='*40}")
    print(f"Training Info: Samples={len(train_dataset)}, Epochs={EPOCHS}")
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'test': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }
    
    # Initialize Model (ResNet18)
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        # We only really care about final performance, but tracking validation is good
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Evaluation Phase
        model.eval()
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_acc = running_corrects.double() / len(test_dataset)
        
        # 记录到TensorBoard
        if writer is not None:
            writer.add_scalar(f'{experiment_name}/Loss', epoch_loss, epoch + writer_step_offset)
            writer.add_scalar(f'{experiment_name}/Accuracy', epoch_acc.item(), epoch + writer_step_offset)
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Test Acc: {epoch_acc:.4f}")
            
    print(f"Best Test Accuracy: {best_acc:.4f}")
    return best_acc



def main():
    os.makedirs(EVAL_RESULT_DIR, exist_ok=True)
    
    # 创建TensorBoard日志目录（保存在项目根目录下的runs/Evaluate_gan_augmentation目录）
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    runs_dir = os.path.join(project_root, "runs")
    eval_folder_name = "Evaluate_gan_augmentation"
    log_dir = os.path.join(runs_dir, eval_folder_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)
    print(f"\nTensorBoard logs will be saved to: {log_dir}")
    print("To view the logs, run: tensorboard --logdir=" + log_dir)
    print(f"{'='*60}")
    
    # 1. Load Real Data
    real_train_ds, real_test_ds, num_classes, class_names = load_real_data()
    
    # 2. Train Baseline (Real Only)
    baseline_acc = train_classifier(real_train_ds, real_test_ds, num_classes, "Baseline (Real Only)", writer, writer_step_offset=0)
    
    # 3. Generate Fake Data
    # Calculate how many to generate
    # augment_ratio * len(real_train) / num_classes = images per class
    target_total_fake = int(len(real_train_ds) * AUGMENT_RATIO)
    imgs_per_class = target_total_fake // num_classes
    
    fake_dataset = generate_fake_data(
        imgs_per_class, 
        num_classes, 
        GENERATOR_CHECKPOINT, 
        GENERATOR_TYPE, 
        DEVICE
    )
    
    # 4. Construct Augmented Dataset
    augmented_train_ds = ConcatDataset([real_train_ds, fake_dataset])
    print(f"\nAugmented Dataset Created: {len(real_train_ds)} Real + {len(fake_dataset)} Fake = {len(augmented_train_ds)} Total")
    
    # 5. Train Experiment (Real + Fake)
    aug_acc = train_classifier(augmented_train_ds, real_test_ds, num_classes, "Experiment (Real + Fake)", writer, writer_step_offset=EPOCHS)
    
    # 记录最终结果到TensorBoard
    writer.add_scalar("Final_Results/Baseline_Accuracy", baseline_acc.item() if hasattr(baseline_acc, 'item') else baseline_acc)
    writer.add_scalar("Final_Results/Augmented_Accuracy", aug_acc.item() if hasattr(aug_acc, 'item') else aug_acc)
    writer.add_scalar("Final_Results/Performance_Gain", (aug_acc - baseline_acc).item() if hasattr(aug_acc - baseline_acc, 'item') else (aug_acc - baseline_acc))
    
    # 关闭SummaryWriter
    writer.close()
    
    # 6. Report
    print(f"\n{'#'*40}")
    print(f"FINAL EVALUATION REPORT")
    print(f"{'#'*40}")
    print(f"Generator Evaluated: {GENERATOR_CHECKPOINT}")
    print(f"Augmentation Ratio: {AUGMENT_RATIO}")
    print(f"{'-'*40}")
    print(f"Baseline Accuracy (Real Only): {baseline_acc:.4f}")
    print(f"Augmented Accuracy (Real + Fake):   {aug_acc:.4f}")
    print(f"{'-'*40}")
    delta = aug_acc - baseline_acc
    print(f"Performance Gain: {delta:+.4f} ({delta*100:+.2f}%)")
    
    # Save result to file
    # 将报告文件也保存到与TensorBoard日志相同的目录
    report_file = os.path.join(log_dir, "augmentation_report.txt")
    with open(report_file, "a") as f:
        f.write(f"\n--- Evaluation at {time.ctime()} ---")
        f.write(f"Generator: {GENERATOR_CHECKPOINT}\n")
        f.write(f"Baseline: {baseline_acc:.4f} | Augmented: {aug_acc:.4f} | Delta: {delta:+.4f}\n")
    
    print(f"\n{'='*40}")
    print("Evaluation Complete!")
    print(f"Results and TensorBoard logs saved to: {log_dir}")
    print("\nTo view the training curves:")
    print("1. Open a terminal")
    print(f"2. Run: tensorboard --logdir={log_dir}")
    print("3. Open your browser and go to: http://localhost:6006")
    print(f"\n{'='*40}")

if __name__ == "__main__":
    main()
