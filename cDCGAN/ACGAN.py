
import os
import random
import shutil
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from pytorch_fid.fid_score import calculate_fid_given_paths


# ============================================================
# Global Hyperparameters
# ============================================================
LATENT_DIM = 100
IMG_SIZE = 64
IMG_CHANNELS = 1
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
EPOCHS = 200
SAVE_PATH = "../Test_Results/Test_ACGAN_Results"

# Reproducibility (set to None to disable)
SEED = 42

# FID settings
FID_EVERY_N_EPOCHS = 10
FID_NUM_SAMPLES = 500

# Feature Flags
ENABLE_LEE_FILTER = False  # Disabled by default for ACGAN texture experiment


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism can slow things down; enable if you need exact reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_empty_dir(path: str) -> None:
    """Create an empty directory (delete if exists)."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def save_images_to_dir(images: torch.Tensor, directory: str, start_index: int = 0) -> None:
    """Save a batch of images to directory as PNG files."""
    # images: (N, C, H, W)
    for idx, img in enumerate(images):
        save_image(
            img.detach().cpu(),
            os.path.join(directory, f"{start_index + idx}.png"),
            normalize=True,
        )


# ============================================================
# Lee Filter (optional preprocessing)
# ============================================================
def lee_filter(image_tensor: torch.Tensor, kernel_size: int = 7, noise_variance: float = 0.01) -> torch.Tensor:
    """
    Lee filter for speckle noise reduction.
    Expects a tensor shaped (C, H, W) and returns the same shape.
    """
    # Add batch dimension: (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)

    local_mean = F.avg_pool2d(
        image_tensor,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    local_mean_sq = F.avg_pool2d(
        image_tensor * image_tensor,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    local_variance = local_mean_sq - local_mean * local_mean
    weight = local_variance / (local_variance + noise_variance)

    filtered = local_mean + weight * (image_tensor - local_mean)
    return filtered.squeeze(0)


# ============================================================
# Generator (ACGAN)
# ============================================================
# Conceptually similar to cDCGAN generator: takes Noise + Class -> Image
# We keep using the specific label embedding strategy.
class Generator(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        # One-hot-like embedding: num_classes -> num_classes
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
        combined = torch.cat([noise_vec, label_embed], dim=1)  # (N, LATENT_DIM + C)
        combined = combined.unsqueeze(2).unsqueeze(3)          # (N, LATENT_DIM + C, 1, 1)
        return self.net(combined)


# ============================================================
# Discriminator (ACGAN)
# ============================================================
class Discriminator(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        
        self.features = nn.Sequential(
            # Input: (1, 64, 64)
            nn.Conv2d(IMG_CHANNELS, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output layers (The feature map size here is 512 x 4 x 4)
        # ACGAN Head 1: Validity (Real/Fake)
        # 真假判别层
        self.adv_layer = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            # Output: (1, 1, 1, 1) -> squeeze to (1)
        )

        # ACGAN Head 2: Classification (Class probs)
        # 类别判别层
        self.aux_layer = nn.Sequential(
            nn.Conv2d(512, class_count, 4, 1, 0, bias=False),
            # Output: (1, class_count, 1, 1) -> squeeze to (class_count)
        )

    def forward(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(image_tensor)
        #  真假概率和类别预测分别输出
        validity = self.adv_layer(features).view(-1, 1)      # (N, 1)
        label_logits = self.aux_layer(features).view(features.size(0), -1)  # (N, Class_Count)
        
        return validity, label_logits


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
                if saved_count >= num_samples:
                    break

                save_count = min(real_imgs.size(0), num_samples - saved_count)

                # Save real images (to CPU)
                save_images_to_dir(real_imgs[:save_count], real_dir, saved_count)

                # Generate matching number of fake images
                noise_vec = torch.randn(save_count, LATENT_DIM, device=run_device)
                cond_labels = torch.randint(0, num_classes_global, (save_count,), device=run_device)
                fake_imgs = gen_model(noise_vec, cond_labels)

                save_images_to_dir(fake_imgs, fake_dir, saved_count)
                saved_count += save_count

        fid_score = calculate_fid_given_paths(
            [real_dir, fake_dir],
            batch_size=50,
            device=run_device,
            dims=2048,
            num_workers=0,
        )

        print(f">>> Epoch {epoch_index} | FID = {fid_score:.4f}\n")
        return float(fid_score)

    finally:
        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(fake_dir, ignore_errors=True)
        gen_model.train()


# ============================================================
# Training Loop
# ============================================================
# 生成一个长度为 n 的标签序列，标签依次从 0 → num_classes-1 周期性循环
# 用于生成测试展示图
def make_sample_labels(num_classes: int, n: int, device: torch.device) -> torch.Tensor:
    base = torch.arange(num_classes, device=device, dtype=torch.long)
    reps = (n + num_classes - 1) // num_classes  # 向下取整除法，计算循环次数
    labels = base.repeat(reps)[:n]
    return labels

# 权重初始化
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

    run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {run_device}")

    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Conditional Preprocessing
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
    dataset = ImageFolder(
        root=dataset_root,
        transform=transform_pipeline,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(run_device.type == "cuda"),
    )

    num_classes_global = len(dataset.classes)
    print(f"Number of classes: {num_classes_global}")

    # Initialize Models
    gen_model = Generator(num_classes_global).to(run_device)
    disc_model = Discriminator(num_classes_global).to(run_device)
    
    gen_model.apply(weights_init_normal)
    disc_model.apply(weights_init_normal)

    # Loss Functions
    adversarial_loss = nn.BCEWithLogitsLoss()
    auxiliary_loss = nn.CrossEntropyLoss()

    optimizer_g = optim.Adam(gen_model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(disc_model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # Tensorboard
    writer = SummaryWriter(log_dir="../runs/ACGAN")

    for epoch_index in range(EPOCHS):
        for i, (real_imgs, real_labels) in enumerate(train_loader):
            real_imgs = real_imgs.to(run_device, non_blocking=True)
            real_labels = real_labels.to(run_device, non_blocking=True)

            curr_batch_size = real_imgs.size(0)

            # Ground truths
            valid = torch.ones(curr_batch_size, 1, device=run_device, requires_grad=False)
            fake = torch.zeros(curr_batch_size, 1, device=run_device, requires_grad=False)

            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_g.zero_grad()

            # 生成随机噪声向量
            noise_vec = torch.randn(curr_batch_size, LATENT_DIM, device=run_device)
            # 生成随机标签向量：从0到num_classes_global-1，生成curr_batch_size个
            gen_labels = torch.randint(0, num_classes_global, (curr_batch_size,), device=run_device)
            
            gen_imgs = gen_model(noise_vec, gen_labels)

            # ACGAN Generator Loss:
            # 1. Adversarial: D should think generated images are Real (valid=1)
            # 2. Auxiliary: D should classify generated images as gen_labels
            validity, pred_label = disc_model(gen_imgs)
            
            loss_g_adv = adversarial_loss(validity, valid)
            loss_g_aux = auxiliary_loss(pred_label, gen_labels)
            
            # Weighted Sum (can tune weights, standard is 1:1)
            loss_g = loss_g_adv + loss_g_aux
            
            loss_g.backward()
            optimizer_g.step()

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_d.zero_grad()

            # Real Loss
            real_pred, real_aux = disc_model(real_imgs)
            loss_d_real_adv = adversarial_loss(real_pred, valid)
            loss_d_real_aux = auxiliary_loss(real_aux, real_labels)
            
            loss_d_real = loss_d_real_adv + loss_d_real_aux

            # Fake Loss
            fake_pred, fake_aux = disc_model(gen_imgs.detach())
            loss_d_fake_adv = adversarial_loss(fake_pred, fake)
            loss_d_fake_aux = auxiliary_loss(fake_aux, gen_labels)
            
            loss_d_fake = loss_d_fake_adv + loss_d_fake_aux
            
            loss_d = (loss_d_real + loss_d_fake) / 2
            
            loss_d.backward()
            optimizer_d.step()
            
            # Calculate classification accuracy for monitoring
            pred_class = torch.argmax(real_aux.data, dim=1)
            acc = (pred_class == real_labels).float().mean()

        print(
            f"[Epoch {epoch_index + 1}/{EPOCHS}] "
            f"Loss_D: {loss_d.item():.4f} | Loss_G: {loss_g.item():.4f} | "
            f"Acc_Real: {acc.item() * 100:.2f}%"
        )
        writer.add_scalar("Loss/Discriminator", loss_d.item(), epoch_index + 1)
        writer.add_scalar("Loss/Generator", loss_g.item(), epoch_index + 1)
        writer.add_scalar("Accuracy/Real", acc.item(), epoch_index + 1)

        # Save samples & compute FID
        if (epoch_index + 1) % FID_EVERY_N_EPOCHS == 0:
            gen_model.eval()
            with torch.no_grad():
                noise_vec = torch.randn(64, LATENT_DIM, device=run_device)
                sample_labels = make_sample_labels(num_classes_global, 64, run_device)
                samples = gen_model(noise_vec, sample_labels)

            save_image(
                samples.detach().cpu(),
                os.path.join(SAVE_PATH, f"epoch_{epoch_index + 1}.png"),
                nrow=8,
                normalize=True,
            )

            fid_score = calculate_fid_score(
                gen_model,
                train_loader,
                epoch_index + 1,
                run_device,
                num_classes_global,
                num_samples=FID_NUM_SAMPLES,
            )
            writer.add_scalar("FID", fid_score, epoch_index + 1)
            
            gen_model.train()

    print("Training completed.")
    # Save final models
    torch.save(gen_model.state_dict(), os.path.join(SAVE_PATH, "generator_final.pth"))
    torch.save(disc_model.state_dict(), os.path.join(SAVE_PATH, "discriminator_final.pth"))


if __name__ == "__main__":
    train()
