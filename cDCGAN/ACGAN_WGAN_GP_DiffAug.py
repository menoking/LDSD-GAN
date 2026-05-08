
import os
import random
import shutil
import numpy as np
from typing import Tuple

import torch
from torch import nn, optim
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import torch.nn.functional as F

from pytorch_fid.fid_score import calculate_fid_given_paths
from DiffAugment_pytorch import DiffAugment


# ============================================================
# Global Hyperparameters
# ============================================================
LATENT_DIM = 100
IMG_SIZE = 64
IMG_CHANNELS = 1
BATCH_SIZE = 64
LEARNING_RATE = 1e-4 
EPOCHS = 200
SAVE_PATH = "../Test_Results/Test_ACGAN_WGAN_GP_DiffAug_Results"

# WGAN-GP Specific
LAMBDA_GP = 10       
N_CRITIC = 3         # Reduced from 5 for better G/D balance
LAMBDA_AUX = 2.5     # Increased from 1.0 to fix Class 0&4 collapse
LABEL_SMOOTHING = 0.1  # Label smoothing for auxiliary loss         

# DiffAugment Policy
# 'color': Brightness/Contrast/Saturation (Sat disabled for grayscale)
# 'translation': Random shifts
# 'cutout': Random masking (Removed for SAR as it might hide small targets)
DIFFAUG_POLICY = 'color,translation'

# Reproducibility
SEED = 42

# FID settings
FID_EVERY_N_EPOCHS = 10
FID_NUM_SAMPLES = 500

# Feature Flags
ENABLE_LEE_FILTER = False


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_empty_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def save_images_to_dir(images: torch.Tensor, directory: str, start_index: int = 0) -> None:
    for idx, img in enumerate(images):
        save_image(
            img.detach().cpu(),
            os.path.join(directory, f"{start_index + idx}.png"),
            normalize=True,
        )


def lee_filter(image_tensor: torch.Tensor, kernel_size: int = 7, noise_variance: float = 0.01) -> torch.Tensor:
    image_tensor = image_tensor.unsqueeze(0)
    local_mean = F.avg_pool2d(image_tensor, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    local_mean_sq = F.avg_pool2d(image_tensor * image_tensor, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    local_variance = local_mean_sq - local_mean * local_mean
    weight = local_variance / (local_variance + noise_variance)
    filtered = local_mean + weight * (image_tensor - local_mean)
    return filtered.squeeze(0)


# ============================================================
# Generator (ACGAN - WGAN Compatible)
# ============================================================
class Generator(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        self.label_embedding = nn.Embedding(class_count, class_count)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM + class_count, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, IMG_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise_vec: torch.Tensor, cond_labels: torch.Tensor) -> torch.Tensor:
        label_embed = self.label_embedding(cond_labels)
        combined = torch.cat([noise_vec, label_embed], dim=1)
        combined = combined.unsqueeze(2).unsqueeze(3)
        return self.net(combined)


# ============================================================
# Discriminator (ACGAN - WGAN-GP Version)
# ============================================================
class Discriminator(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        
        self.features = nn.Sequential(
            # Input: (1, 64, 64)
            nn.Conv2d(IMG_CHANNELS, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128, affine=True), 
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512, affine=True), 
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.adv_layer = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )
        
        self.aux_layer = nn.Sequential(
            nn.Conv2d(512, class_count, 4, 1, 0, bias=False),
        )

    def forward(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(image_tensor)
        
        validity = self.adv_layer(features).view(-1, 1)      
        label_logits = self.aux_layer(features).view(features.size(0), -1) 
        
        return validity, label_logits


# ============================================================
# Gradient Penalty
# ============================================================
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # NOTE: D is passed augmented images in training, but GP should ideally be consistent.
    # However, since D learns on augmented data, we should also compute GP on augmented interpolates?
    # Or just raw? Standard implementations usually apply Augment inside D or apply to real/fake before GP.
    # Here we assume real_samples and fake_samples are ALREADY augmented.
    
    d_interpolates, _ = D(interpolates) 
    
    fake = torch.ones(d_interpolates.shape, device=device, requires_grad=False)
    
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


# ============================================================
# FID Calculation
# ============================================================
def calculate_fid_score(
    gen_model: nn.Module,
    train_loader: DataLoader,
    epoch_index: int,
    run_device: torch.device,
    num_classes_global: int,
    num_samples: int = FID_NUM_SAMPLES,
) -> float:
    print(f"\n[FID] Evaluating Epoch {epoch_index}")
    real_dir = "./fid_real"
    fake_dir = "./fid_fake"
    ensure_empty_dir(real_dir)
    ensure_empty_dir(fake_dir)
    gen_model.eval()
    saved_count = 0
    try:
        with torch.no_grad():
            for real_imgs, _ in train_loader:
                if saved_count >= num_samples: break
                save_count = min(real_imgs.size(0), num_samples - saved_count)
                save_images_to_dir(real_imgs[:save_count], real_dir, saved_count)
                noise_vec = torch.randn(save_count, LATENT_DIM, device=run_device)
                cond_labels = torch.randint(0, num_classes_global, (save_count,), device=run_device)
                fake_imgs = gen_model(noise_vec, cond_labels)
                save_images_to_dir(fake_imgs, fake_dir, saved_count)
                saved_count += save_count
        fid_score = calculate_fid_given_paths([real_dir, fake_dir], batch_size=50, device=run_device, dims=2048, num_workers=0)
        print(f">>> Epoch {epoch_index} | FID = {fid_score:.4f}\n")
        return float(fid_score)
    finally:
        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(fake_dir, ignore_errors=True)
        gen_model.train()


# ============================================================
# Training Loop
# ============================================================
def check_cuda_availability():
    if torch.cuda.is_available():
        print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("CUDA is NOT available. Using CPU. Warning: Training will be extremely slow.")
        return torch.device("cpu")

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1: 
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train() -> None:
    if SEED is not None:
        set_seed(SEED)

    run_device = check_cuda_availability()

    os.makedirs(SAVE_PATH, exist_ok=True)
    
    transforms_list = [
        transforms.Grayscale(1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
    if ENABLE_LEE_FILTER:
        transforms_list.append(transforms.Lambda(lambda x: lee_filter(x)))
    transforms_list.append(transforms.Normalize((0.5,), (0.5,)))

    transform_pipeline = transforms.Compose(transforms_list)

    dataset_root = os.path.join("..", "MSTAR", "PERSONAL_MSTAR", "15_DEG")
    dataset = ImageFolder(root=dataset_root, transform=transform_pipeline)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(run_device.type=="cuda"))

    num_classes_global = len(dataset.classes)
    print(f"Number of classes: {num_classes_global}")

    # Initialize Models
    gen_model = Generator(num_classes_global).to(run_device)
    disc_model = Discriminator(num_classes_global).to(run_device)
    
    gen_model.apply(weights_init_normal)
    disc_model.apply(weights_init_normal)

    optimizer_g = optim.Adam(gen_model.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    optimizer_d = optim.Adam(disc_model.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    
    auxiliary_loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    print(f"DiffAugment Policy: {DIFFAUG_POLICY}")

    for epoch_index in range(EPOCHS):
        for i, (real_imgs, real_labels) in enumerate(train_loader):
            real_imgs = real_imgs.to(run_device)
            real_labels = real_labels.to(run_device)
            curr_batch_size = real_imgs.size(0)

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_d.zero_grad()

            # Fake - Use balanced class sampling to prevent mode collapse
            noise_vec = torch.randn(curr_batch_size, LATENT_DIM, device=run_device)
            # Ensure all classes are represented in each batch
            samples_per_class = curr_batch_size // num_classes_global
            gen_labels = torch.cat([
                torch.full((samples_per_class,), i, dtype=torch.long, device=run_device)
                for i in range(num_classes_global)
            ])
            # Fill remaining slots if batch size not divisible by num_classes
            if len(gen_labels) < curr_batch_size:
                remaining = curr_batch_size - len(gen_labels)
                gen_labels = torch.cat([gen_labels, torch.randint(0, num_classes_global, (remaining,), device=run_device)])
            gen_labels = gen_labels[:curr_batch_size]
            fake_imgs = gen_model(noise_vec, gen_labels)
            
            # --- APPLY DiffAugment ---
            real_imgs_aug = DiffAugment(real_imgs, policy=DIFFAUG_POLICY)
            fake_imgs_aug = DiffAugment(fake_imgs.detach(), policy=DIFFAUG_POLICY)
            
            # --- D Update Strategy: Hybrid DiffAugment ---
            
            # 1. Adversarial Loss Use Augmented images
            real_validity_aug, _ = disc_model(real_imgs_aug)
            fake_validity_aug, _ = disc_model(fake_imgs_aug)
            
            # WGAN Loss
            loss_d_adv = -torch.mean(real_validity_aug) + torch.mean(fake_validity_aug)
            
            # Gradient Penalty (Use Augmented samples)
            gradient_penalty = compute_gradient_penalty(disc_model, real_imgs_aug, fake_imgs_aug, run_device)
            
            # 2. Aux Loss: Use Clean images
            # Need to pass clean real/fake to D
            _, real_aux_logits_clean = disc_model(real_imgs)
            _, fake_aux_logits_clean = disc_model(fake_imgs.detach())

            loss_d_aux = auxiliary_loss(real_aux_logits_clean, real_labels) + \
                         auxiliary_loss(fake_aux_logits_clean, gen_labels)
            
            # Weighted Sum with auxiliary loss weight
            loss_d = loss_d_adv + LAMBDA_GP * gradient_penalty + LAMBDA_AUX * loss_d_aux
            
            # Calculate classification accuracy for monitoring
            with torch.no_grad():
                pred_real = torch.argmax(real_aux_logits_clean, dim=1)
                pred_fake = torch.argmax(fake_aux_logits_clean, dim=1)
                acc_real = (pred_real == real_labels).float().mean()
                acc_fake = (pred_fake == gen_labels).float().mean()
            
            loss_d.backward()
            optimizer_d.step()

            # ---------------------
            # Train Generator (Every n_critic steps)
            # ---------------------
            if i % N_CRITIC == 0:
                optimizer_g.zero_grad()
                
                # Regenerate fakes for G update - use balanced sampling
                noise_vec = torch.randn(curr_batch_size, LATENT_DIM, device=run_device)
                samples_per_class = curr_batch_size // num_classes_global
                gen_labels = torch.cat([
                    torch.full((samples_per_class,), i, dtype=torch.long, device=run_device)
                    for i in range(num_classes_global)
                ])
                if len(gen_labels) < curr_batch_size:
                    remaining = curr_batch_size - len(gen_labels)
                    gen_labels = torch.cat([gen_labels, torch.randint(0, num_classes_global, (remaining,), device=run_device)])
                gen_labels = gen_labels[:curr_batch_size]
                gen_imgs = gen_model(noise_vec, gen_labels)
                
                # --- G Update Strategy: Hybrid DiffAugment ---
                # 1. Adversarial Loss: Use Augmented images (Robustness)
                gen_imgs_aug = DiffAugment(gen_imgs, policy=DIFFAUG_POLICY)
                fake_validity_aug, _ = disc_model(gen_imgs_aug)
                loss_g_adv = -torch.mean(fake_validity_aug)
                
                # 2. Aux Loss: Use Clean images (Correct Class Features)
                # Note: We re-pass clean images to D to get clean logits
                _, fake_aux_logits_clean = disc_model(gen_imgs)
                loss_g_aux = auxiliary_loss(fake_aux_logits_clean, gen_labels)
                
                # Weighted Sum with auxiliary loss weight
                loss_g = loss_g_adv + LAMBDA_AUX * loss_g_aux
                
                loss_g.backward()
                optimizer_g.step()

        # Monitoring
        print(f"[Epoch {epoch_index + 1}/{EPOCHS}] D_Loss: {loss_d.item():.4f} | G_Loss: {loss_g.item():.4f} | Acc_Real: {acc_real.item():.3f} | Acc_Fake: {acc_fake.item():.3f}")

        # Save & FID
        if (epoch_index + 1) % FID_EVERY_N_EPOCHS == 0:
            gen_model.eval()
            with torch.no_grad():
                noise_vec = torch.randn(64, LATENT_DIM, device=run_device)
                # Correct Fix from previous debugging
                sample_labels = torch.arange(num_classes_global, device=run_device).repeat(20)[:64]
                samples = gen_model(noise_vec, sample_labels)
            
            save_image(
                samples.detach().cpu(),
                os.path.join(SAVE_PATH, f"epoch_{epoch_index + 1}.png"),
                nrow=8,
                normalize=True,
            )
            
            # Note: FID is calculated on NON-augmented images (Standard practice)
            calculate_fid_score(gen_model, train_loader, epoch_index + 1, run_device, num_classes_global)
            gen_model.train()

    print("Training completed.")
    torch.save(gen_model.state_dict(), os.path.join(SAVE_PATH, "generator_final.pth"))
    torch.save(disc_model.state_dict(), os.path.join(SAVE_PATH, "discriminator_final.pth"))


if __name__ == "__main__":
    train()
