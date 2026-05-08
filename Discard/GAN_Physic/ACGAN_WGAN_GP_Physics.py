import os
import random
import shutil
import numpy as np
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from pytorch_fid.fid_score import calculate_fid_given_paths

# ============================================================
# Global Hyperparameters
# ============================================================
LATENT_DIM = 100
IMG_SIZE = 64
IMG_CHANNELS = 1
BATCH_SIZE = 64
LEARNING_RATE = 1e-4  # WGAN-GP usually uses lower LR
EPOCHS = 100
SAVE_PATH = "../Test_Results/Test_ACGAN_WGAN_GP_Physics_Results"

# WGAN-GP Specific
LAMBDA_GP = 10
N_CRITIC = 3         # Reduced from 5 for better G/D balance
LAMBDA_AUX = 3.0     # Increased to 3.0 - physics noise needs strong class guidance
LABEL_SMOOTHING = 0.05  # Label smoothing to prevent overconfidence

# Physics parameters (Speckle noise)
# SAR speckle follows a Gamma distribution with shape = L
# L=5.0 is more realistic for multi-looked processed SAR data (e.g. MSTAR)
SPECKLE_SHAPE = 5.0

# FID settings
FID_EVERY_N_EPOCHS = 10
FID_NUM_SAMPLES = 500

# Reproducibility
SEED = 42


# ============================================================
# Utilities
# ============================================================
def save_images_to_dir(images: torch.Tensor, directory: str, start_index: int = 0) -> None:
    for idx, img in enumerate(images):
        save_image(
            img.detach().cpu(),
            os.path.join(directory, f"{start_index + idx}.png"),
            normalize=True,
        )


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
        fid_score = calculate_fid_given_paths([real_dir, fake_dir], batch_size=50, device=run_device, dims=2048,
                                              num_workers=0)
        print(f">>> Epoch {epoch_index} | FID = {fid_score:.4f}\n")
        return float(fid_score)
    finally:
        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(fake_dir, ignore_errors=True)
        gen_model.train()


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


def speckle_noise(batch: int, device: torch.device) -> torch.Tensor:
    scale = 1.0 / SPECKLE_SHAPE
    noise = torch.distributions.Gamma(SPECKLE_SHAPE, scale).sample((batch, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)).to(device)
    return noise


# ============================================================
# Generator (ACGAN + Physics)
# ============================================================
class Generator(nn.Module):
    def __init__(self, class_count: int):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(class_count, class_count)

        self.net = nn.Sequential(
            # 1. First Dense layer to 4x4
            nn.ConvTranspose2d(LATENT_DIM + class_count, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 2. Upsample + Conv to 8x8
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 3. Upsample + Conv to 16x16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 4. Upsample + Conv to 32x32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 5. Upsample + Conv to 64x64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, IMG_CHANNELS, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)
        gen_input = gen_input.view(gen_input.size(0), -1, 1, 1)

        # 1. Output clean reflectance [-1, 1]
        reflectance_raw = self.net(gen_input)

        # 2. Rescale to [0, 1] for physics-informed multiplication
        reflectance = (reflectance_raw + 1) / 2

        # 3. Apply Multiplicative Speckle Noise (Physics)
        noise_mask = speckle_noise(noise.size(0), noise.device)
        fake_imgs_raw = reflectance * noise_mask

        # 3.5 Clamp to [0, 1] to prevent values > 1 from leaking to Discriminator
        fake_imgs = torch.clamp(fake_imgs_raw, 0, 1)

        # 4. Rescale back to [-1, 1] for training stability / Discriminator
        return (fake_imgs * 2) - 1


# ============================================================
# Discriminator (ACGAN-style)
# ============================================================
class Discriminator(nn.Module):
    def __init__(self, class_count: int):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            # Input: (1, 64, 64)
            nn.Conv2d(IMG_CHANNELS, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),  # WGAN-GP uses InstanceNorm
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output layers
        self.adv_layer = nn.Sequential(nn.Conv2d(512, 1, 4, 1, 0))  # Real/Fake
        self.aux_layer = nn.Sequential(nn.Conv2d(512, class_count, 4, 1, 0))  # Classification

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(img)
        validity = self.adv_layer(feat).view(-1, 1)
        label_logits = self.aux_layer(feat).view(img.size(0), -1)
        return validity, label_logits


# ============================================================
# WGAN-GP Utils
# ============================================================
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    gradients = torch.autograd.grad(
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
# Training
# ============================================================
def train():
    global fake_imgs, loss_D, loss_G
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_PATH, exist_ok=True)

    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = ImageFolder(root="../MSTAR/PERSONAL_MSTAR/15_DEG", transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    class_count = len(dataset.classes)

    generator = Generator(class_count).to(device)
    discriminator = Discriminator(class_count).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))

    # Tensorboard
    writer = SummaryWriter(log_dir="../runs/ACGAN_WGAN_GP_Physics")

    aux_loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    print(f"Starting ACGAN-WGAN-GP-Physics Training on {device}...")

    for epoch in range(EPOCHS):
        for i, (real_imgs, labels) in enumerate(dataloader):
            real_imgs, labels = real_imgs.to(device), labels.to(device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Use balanced class sampling to prevent mode collapse
            noise = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
            samples_per_class = BATCH_SIZE // class_count
            gen_labels = torch.cat([
                torch.full((samples_per_class,), i, dtype=torch.long, device=device)
                for i in range(class_count)
            ])
            if len(gen_labels) < BATCH_SIZE:
                remaining = BATCH_SIZE - len(gen_labels)
                gen_labels = torch.cat([gen_labels, torch.randint(0, class_count, (remaining,), device=device)])
            gen_labels = gen_labels[:BATCH_SIZE]
            fake_imgs = generator(noise, gen_labels)

            real_validity, real_aux = discriminator(real_imgs)
            fake_validity, fake_aux = discriminator(fake_imgs.detach())

            # WGAN Gradient Penalty
            gp = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, device)

            # Adversarial Loss
            loss_adv = -torch.mean(real_validity) + torch.mean(fake_validity)

            # Auxiliary Loss (Classification) with increased weight
            loss_aux = aux_loss(real_aux, labels) + aux_loss(fake_aux, gen_labels)
            
            # Calculate classification accuracy for monitoring
            with torch.no_grad():
                pred_real = torch.argmax(real_aux, dim=1)
                pred_fake = torch.argmax(fake_aux, dim=1)
                acc_real = (pred_real == labels).float().mean()
                acc_fake = (pred_fake == gen_labels).float().mean()

            loss_D = loss_adv + LAMBDA_GP * gp + LAMBDA_AUX * loss_aux

            loss_D.backward()
            optimizer_D.step()

            # ---------------------
            # Train Generator (Every N_CRITIC steps)
            # ---------------------
            if i % N_CRITIC == 0:
                optimizer_G.zero_grad()

                gen_imgs = generator(noise, gen_labels)
                validity, pred_label = discriminator(gen_imgs)

                loss_g_adv = -torch.mean(validity)
                loss_g_aux = aux_loss(pred_label, gen_labels)

                loss_G = loss_g_adv + LAMBDA_AUX * loss_g_aux

                loss_G.backward()
                optimizer_G.step()

        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch + 1}/{EPOCHS}] D_Loss: {loss_D.item():.4f} | G_Loss: {loss_G.item():.4f} | Acc_Real: {acc_real.item():.3f} | Acc_Fake: {acc_fake.item():.3f}")
            save_image(fake_imgs.data[:25], f"{SAVE_PATH}/epoch_{epoch + 1}.png", nrow=5, normalize=True)

            # FID
            fid_score = calculate_fid_score(generator, dataloader, epoch + 1, device, class_count)
            writer.add_scalar("FID", fid_score, epoch + 1)

        # Log losses every epoch
        writer.add_scalar("Loss/Discriminator", loss_D.item(), epoch + 1)
        writer.add_scalar("Loss/Generator", loss_G.item(), epoch + 1)
        writer.add_scalar("Loss/Gradient_Penalty", gp.item(), epoch + 1)
        writer.add_scalar("Accuracy/Real", acc_real.item(), epoch + 1)
        writer.add_scalar("Accuracy/Fake", acc_fake.item(), epoch + 1)

    print("Training Complete.")
    writer.close()
    torch.save(generator.state_dict(), os.path.join(SAVE_PATH, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(SAVE_PATH, "discriminator_final.pth"))


if __name__ == "__main__":
    train()
