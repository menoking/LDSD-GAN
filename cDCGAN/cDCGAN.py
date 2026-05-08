
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
SAVE_PATH = "../Test_Results/Test_cDCGAN_Results"

# Reproducibility
SEED = 42

# FID settings
FID_EVERY_N_EPOCHS = 10  # 间隔采样轮次
FID_NUM_SAMPLES = 150  # 取500真实样本，500生成样本进行计算

# Feature Flags
ENABLE_LEE_FILTER = False  # cDCGAN usually used Lee filter in previous experiments


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
# Generator (cDCGAN)
# ============================================================
class Generator(nn.Module):
    def __init__(self, class_count: int):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(class_count, class_count)

        self.net = nn.Sequential(
            # Input: LATENT_DIM + class_count
            # 逆卷积膨胀
            nn.ConvTranspose2d(LATENT_DIM + class_count, 512, 4, 1, 0, bias=False),
            # 二维批量归一化
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
        # 由于模型中输入为四维张量，因此unsqueeze在原本张量的第2和3位插入大小为1的新维度
        combined = combined.unsqueeze(2).unsqueeze(3)  # (N, dim, 1, 1)
        return self.net(combined)


# ============================================================
# Discriminator (cDCGAN)
# ============================================================
class Discriminator(nn.Module):
    def __init__(self, class_count: int):
        super(Discriminator, self).__init__()
        # Unlike MLP cGAN, convolutional D usually appends label as a channel or similar.
        # Here we use the approach of embedding labels into an image channel.
        
        self.label_embedding = nn.Embedding(class_count, IMG_SIZE * IMG_SIZE)
        
        self.net = nn.Sequential(
            # 卷积提取特征
            nn.Conv2d(IMG_CHANNELS + 1, 64, 4, 2, 1, bias=False),
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
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image_tensor: torch.Tensor, cond_labels: torch.Tensor) -> torch.Tensor:
        # 这里不要搞混了，其在构造函数中已定义，因此只需要传入参数cond_labels后即可输出IMG_SIZE * IMG_SIZE的张量
        label_embed = self.label_embedding(cond_labels)
        # -1自动推断batch size，其余参数依次为CHW
        label_img = label_embed.view(-1, 1, IMG_SIZE, IMG_SIZE)

        # Concatenate image and label channel
        combined = torch.cat([image_tensor, label_img], dim=1)

        return self.net(combined).view(-1, 1)


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
                random_labels = torch.randint(0, num_classes_global, (save_count,), device=run_device)
                fake_imgs = gen_model(noise_vec, random_labels)
                save_images_to_dir(fake_imgs, fake_dir, saved_count)
                saved_count += save_count
                
        fid_score = calculate_fid_given_paths([real_dir, fake_dir], batch_size=50, device=run_device, dims=2048, num_workers=0)
        print(f">>> Epoch {epoch_index} | FID = {fid_score:.4f}\n")
        return float(fid_score)
    except Exception as e:
        print(f"FID Error: {e}")
        return 999.0
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

    transform = transforms.Compose(transforms_list)

    dataset_root = "../MSTAR/PERSONAL_MSTAR/15_DEG"
    train_dataset = ImageFolder(root=dataset_root, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(run_device.type=="cuda"))

    num_classes_global = len(train_dataset.classes)
    print(f"Number of classes: {num_classes_global}")

    # Models
    generator = Generator(num_classes_global).to(run_device)
    discriminator = Discriminator(num_classes_global).to(run_device)
    
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Loss & Optimizers
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # Tensorboard
    writer = SummaryWriter(log_dir="../runs/cDCGAN")

    print("Beginning Training (cDCGAN)...")

    for epoch in range(EPOCHS):
        for i, (real_imgs, labels) in enumerate(train_loader):
            curr_batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(run_device)
            labels = labels.to(run_device)

            real_label = torch.ones(curr_batch_size, 1).to(run_device)
            fake_label = torch.zeros(curr_batch_size, 1).to(run_device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            real_output = discriminator(real_imgs, labels)
            loss_d_real = criterion(real_output, real_label)

            z = torch.randn(curr_batch_size, LATENT_DIM, device=run_device)
            fake_imgs = generator(z, labels)
            fake_output = discriminator(fake_imgs.detach(), labels)
            loss_d_fake = criterion(fake_output, fake_label)

            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            optimizer_D.step()

            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            tricked_output = discriminator(fake_imgs, labels)
            loss_g = criterion(tricked_output, real_label)

            loss_g.backward()
            optimizer_G.step()
        
        # Logging
        print(f"[Epoch {epoch + 1}/{EPOCHS}] D_Loss: {loss_d.item():.4f} | G_Loss: {loss_g.item():.4f}")
        writer.add_scalar("Loss/Discriminator", loss_d.item(), epoch + 1)
        writer.add_scalar("Loss/Generator", loss_g.item(), epoch + 1)

        # Save & Eval
        if (epoch + 1) % FID_EVERY_N_EPOCHS == 0:
            with torch.no_grad():
                fixed_noise = torch.randn(64, LATENT_DIM, device=run_device)
                fixed_labels = torch.zeros(64, dtype=torch.long, device=run_device) 
                fake_samples = generator(fixed_noise, fixed_labels)
                save_image(fake_samples, f"{SAVE_PATH}/epoch_{epoch + 1}.png", nrow=8, normalize=True)
            
            fid_score = calculate_fid_score(generator, train_loader, epoch + 1, run_device, num_classes_global)
            writer.add_scalar("FID", fid_score, epoch + 1)

    print("Training completed.")
    writer.close()
    torch.save(generator.state_dict(), os.path.join(SAVE_PATH, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(SAVE_PATH, "discriminator_final.pth"))

    # Final samples
    generator.eval()
    with torch.no_grad():
        test_noise = torch.randn(16, LATENT_DIM, device=run_device)
        test_labels = torch.randint(0, num_classes_global, (16,), device=run_device)
        test_imgs = generator(test_noise, test_labels)
        save_image(test_imgs, f"{SAVE_PATH}/final_samples.png", normalize=True)


if __name__ == "__main__":
    train()
