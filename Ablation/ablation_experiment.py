
import os
import random
import shutil
import argparse
import sys
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
from torch.utils.tensorboard import SummaryWriter
from pytorch_fid.fid_score import calculate_fid_given_paths

# ============================================================
# Path Resolution
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Ensure DiffAugment_pytorch.py is accessible. 
sys.path.append(os.path.join(PROJECT_ROOT, "cDCGAN"))
try:
    from DiffAugment_pytorch import DiffAugment
except ImportError:
    print(f"Warning: DiffAugment_pytorch not found in {os.path.join(PROJECT_ROOT, 'cDCGAN')}.")

# ============================================================
# Global Hyperparameters (Standardized across all configs)
# ============================================0================
LATENT_DIM = 100
IMG_SIZE = 64
IMG_CHANNELS = 1
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE_G = 2e-4 
LEARNING_RATE_D = 2e-4 

# WGAN-GP Specific
LAMBDA_GP = 10
N_CRITIC = 3
LABEL_SMOOTHING = 0.1
DIFFAUG_POLICY = 'color,translation'

# Physics Parameters
INITIAL_SIGMA = 0.05
# Note: Stat-Loss (LAMBDA_PHYS) is EXCLUDED in this ablation as requested.

# FID settings
FID_EVERY_N_EPOCHS = 10
FID_NUM_SAMPLES = 500
SEED = 42

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

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# ============================================================
# Log-Physics Layer (For Config C)
# ============================================================
class LogPhysicsLayer(nn.Module):
    def __init__(self, num_classes, init_sigma=INITIAL_SIGMA):
        super(LogPhysicsLayer, self).__init__()
        val = np.log(init_sigma)
        self.log_sigma = nn.Parameter(torch.full((num_classes, 1, 1, 1), val))

    def forward(self, log_reflectance, labels):
        batch_log_sigma = self.log_sigma[labels]
        sigma = torch.exp(batch_log_sigma)
        noise = torch.randn_like(log_reflectance) * sigma
        return log_reflectance + noise

# ============================================================
# Generator (Configurable)
# ============================================================
class Generator(nn.Module):
    def __init__(self, class_count: int, config: str):
        super(Generator, self).__init__()
        self.config = config
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
        )
        
        if config == 'C':
            self.physics_layer = LogPhysicsLayer(num_classes=class_count)
        
        self.final_act = nn.Tanh()

    def forward(self, noise_vec, cond_labels):
        label_embed = self.label_embedding(cond_labels)
        combined = torch.cat([noise_vec, label_embed], dim=1)
        combined = combined.unsqueeze(2).unsqueeze(3)
        
        raw_output = self.net(combined)
        
        if self.config == 'A':
            # Linear Domain: Direct Tanh
            return self.final_act(raw_output)
        elif self.config == 'B':
            # Log-Domain Only: No Physics Layer
            return self.final_act(raw_output)
        elif self.config == 'C':
            # Full Physics: Reflectance + Noise
            noisy_log = self.physics_layer(raw_output, cond_labels)
            return self.final_act(noisy_log)
        return raw_output

# ============================================================
# Discriminator (Shared Architecture)
# ============================================================
class Discriminator(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        self.label_embedding = nn.Embedding(class_count, IMG_SIZE * IMG_SIZE)
        self.features = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS + 1, 64, 4, 2, 1, bias=False),
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
        self.adv_layer = nn.Sequential(nn.Conv2d(512, 1, 4, 1, 0, bias=False))

    def forward(self, image_tensor, cond_labels):
        label_embed = self.label_embedding(cond_labels)
        label_img = label_embed.view(-1, 1, IMG_SIZE, IMG_SIZE)
        combined = torch.cat([image_tensor, label_img], dim=1)
        features = self.features(combined)
        validity = self.adv_layer(features).view(-1, 1)
        return validity

def compute_gradient_penalty(D, real_samples, fake_samples, labels, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = torch.ones(d_interpolates.shape, device=device, requires_grad=False)
    gradients = autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def calculate_fid_score(gen_model, train_loader, run_device, num_classes, num_samples):
    real_dir = os.path.join(PROJECT_ROOT, "Ablation", "fid_real")
    fake_dir = os.path.join(PROJECT_ROOT, "Ablation", "fid_fake")
    ensure_empty_dir(real_dir); ensure_empty_dir(fake_dir)
    gen_model.eval()
    saved_count = 0
    with torch.no_grad():
        for real_imgs, _ in train_loader:
            if saved_count >= num_samples: break
            save_count = min(real_imgs.size(0), num_samples - saved_count)
            save_images_to_dir(real_imgs[:save_count], real_dir, saved_count)
            z = torch.randn(save_count, LATENT_DIM, device=run_device)
            labels = torch.randint(0, num_classes, (save_count,), device=run_device)
            fake_imgs = gen_model(z, labels)
            save_images_to_dir(fake_imgs, fake_dir, saved_count)
            saved_count += save_count
    fid_score = calculate_fid_given_paths([real_dir, fake_dir], batch_size=50, device=run_device, dims=2048, num_workers=0)
    shutil.rmtree(real_dir); shutil.rmtree(fake_dir)
    gen_model.train()
    return float(fid_score)

def train(cfg_choice: str) -> None:
    set_seed(SEED)
    run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Path Configuration
    standard_names = {'A': 'A_Baseline', 'B': 'B_LogOnly', 'C': 'C_FullPhysics'}
    config_name = standard_names[cfg_choice]
    save_path = os.path.join(PROJECT_ROOT, "Test_Results", config_name)
    log_path = os.path.join(PROJECT_ROOT, "runs", config_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Data Loading
    transform_pipeline = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset_root = os.path.join(PROJECT_ROOT, "MSTAR", "PERSONAL_MSTAR", "15_DEG")
    dataset = ImageFolder(root=dataset_root, transform=transform_pipeline)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    num_classes = len(dataset.classes)

    # Model Init
    gen_model = Generator(num_classes, cfg_choice).to(run_device)
    disc_model = Discriminator(num_classes).to(run_device)
    gen_model.apply(weights_init_normal)
    disc_model.apply(weights_init_normal)

    optimizer_g = optim.Adam(gen_model.parameters(), lr=LEARNING_RATE_G, betas=(0.0, 0.9))
    optimizer_d = optim.Adam(disc_model.parameters(), lr=LEARNING_RATE_D, betas=(0.0, 0.9))
    writer = SummaryWriter(log_dir=log_path)

    print(f"Starting Ablation Experiment: {config_name}")
    
    for epoch in range(EPOCHS):
        for i, (real_imgs, real_labels) in enumerate(train_loader):
            real_imgs, real_labels = real_imgs.to(run_device), real_labels.to(run_device)
            batch_size = real_imgs.size(0)

            # --- Train Discriminator ---
            optimizer_d.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM, device=run_device)
            # Balanced sampling for gen_labels
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=run_device)
            fake_imgs = gen_model(z, gen_labels)

            real_imgs_aug = DiffAugment(real_imgs, policy=DIFFAUG_POLICY)
            fake_imgs_aug = DiffAugment(fake_imgs.detach(), policy=DIFFAUG_POLICY)
            
            real_validity = disc_model(real_imgs_aug, real_labels)
            fake_validity = disc_model(fake_imgs_aug, gen_labels)

            loss_d_adv = -torch.mean(real_validity) + torch.mean(fake_validity)
            gp = compute_gradient_penalty(disc_model, real_imgs_aug, fake_imgs_aug, real_labels, run_device)
            loss_d = loss_d_adv + LAMBDA_GP * gp
            loss_d.backward(); optimizer_d.step()

            # --- Train Generator ---
            if i % N_CRITIC == 0:
                optimizer_g.zero_grad()
                gen_imgs = gen_model(z, gen_labels)
                gen_imgs_aug = DiffAugment(gen_imgs, policy=DIFFAUG_POLICY)
                loss_g = -torch.mean(disc_model(gen_imgs_aug, gen_labels))
                loss_g.backward(); optimizer_g.step()

        print(f"[{config_name}] Epoch {epoch+1}/{EPOCHS} | D_Loss: {loss_d.item():.4f} | G_Loss: {loss_g.item():.4f}")
        writer.add_scalar("Loss/D", loss_d.item(), epoch+1)
        writer.add_scalar("Loss/G", loss_g.item(), epoch+1)

        if (epoch + 1) % FID_EVERY_N_EPOCHS == 0:
            fid = calculate_fid_score(gen_model, train_loader, run_device, num_classes, FID_NUM_SAMPLES)
            print(f">>> Epoch {epoch+1} | FID: {fid:.4f}")
            writer.add_scalar("Metrics/FID", fid, epoch+1)
            # Save Sample
            fixed_noise = torch.randn(64, LATENT_DIM, device=run_device)
            fixed_labels = torch.arange(num_classes, device=run_device).repeat((64 // num_classes) + 1)[:64]
            with torch.no_grad():
                samples = gen_model(fixed_noise, fixed_labels)
            save_image(samples, os.path.join(save_path, f"epoch_{epoch+1}.png"), nrow=8, normalize=True)

    torch.save(gen_model.state_dict(), os.path.join(save_path, "generator_final.pth"))
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=['A', 'B', 'C'], required=True, help="A: Baseline, B: LogOnly, C: FullPhysics")
    args = parser.parse_args()
    train(args.config)
