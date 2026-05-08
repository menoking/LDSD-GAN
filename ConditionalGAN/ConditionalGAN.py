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
IMAGE_FLAT_SIZE = IMG_CHANNELS * IMG_SIZE * IMG_SIZE
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
EPOCHS = 200  # Standardized to match others usually
SAVE_PATH = "../Test_Results/Test_cGAN_Results"

# Reproducibility
SEED = 42

# FID settings
FID_EVERY_N_EPOCHS = 10  # 10epoch进行一次FID计算；且保存一次图片
FID_NUM_SAMPLES = 500


# 设置随机种子函数
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 设置GPU 上的随机数生成器
    torch.backends.cudnn.deterministic = True  # cuDNN使用确定性算法，放弃随机优化算法
    torch.backends.cudnn.benchmark = False  # 固定benchmark优化，使用固定卷积结构


# 创建文件夹：存在则清空，不存在则创建
def ensure_empty_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


# 图片保存：张量->png
def save_images_to_dir(images: torch.Tensor, directory: str, start_index: int = 0) -> None:
    for idx, img in enumerate(images):  # enumerate内置函数，同时获取索引与键值
        save_image(
            img.detach().cpu(),  # detach()用来分离计算图
            os.path.join(directory, f"{start_index + idx}.png"),
            normalize=True,  # 归一化
        )


# Generator (Conditional MLP)
class Generator(nn.Module):
    def __init__(self, n_classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + n_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, IMAGE_FLAT_SIZE),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        # Concatenate noise and label embedding
        x = torch.cat([z, c], dim=1)
        img = self.model(x)
        return img.view(img.size(0), IMG_CHANNELS, IMG_SIZE, IMG_SIZE)


# Discriminator (Conditional MLP)
class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            # Input is (Image + Label Embedding)
            nn.Linear(IMAGE_FLAT_SIZE + n_classes, 1024),
            nn.LeakyReLU(0.2),
            # Standard cGAN usually has dropout in D.
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        c = self.label_emb(labels)
        # Concatenate image and label embedding
        x = torch.cat([img_flat, c], dim=1)
        validity = self.model(x)
        return validity


# FID Calculation
def calculate_fid_score(
        gen_model: nn.Module,
        train_loader: DataLoader,
        epoch_index: int,
        run_device: torch.device,
        num_classes_global: int,  # 条件标签类别数
        num_samples: int = FID_NUM_SAMPLES,  # 样本个数
) -> float:
    print(f"\n[FID] Evaluating Epoch {epoch_index}")
    real_dir = "./fid_real"
    fake_dir = "./fid_fake"
    ensure_empty_dir(real_dir)
    ensure_empty_dir(fake_dir)
    gen_model.eval()  # 评估模式
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
        # 清除临时文件夹
        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(fake_dir, ignore_errors=True)
        gen_model.train()  # 返回训练模式


# Check whether the device is cuda
def check_cuda_availability():
    if torch.cuda.is_available():
        print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("CUDA is NOT available. Using CPU. Warning: Training will be extremely slow.")
        return torch.device("cpu")


# Training Loop
def train() -> None:
    # 检查随机数种子
    if SEED is not None:
        set_seed(SEED)

    # 检查设备以及存储路径
    run_device = check_cuda_availability()
    os.makedirs(SAVE_PATH, exist_ok=True)

    # 数据预处理：
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转化为单通道
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 缩放到IMG_SIZE*IMG_SIZE
        transforms.ToTensor(),  # 张量化
        transforms.Normalize((0.5,), (0.5,))  # 输入参数是序列，因此用单元素元组；(0,1)->(-1,1)
    ])

    # 数据集加载
    dataset_root = "../MSTAR/PERSONAL_MSTAR/15_DEG"
    train_dataset = ImageFolder(root=dataset_root, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                              pin_memory=(run_device.type == "cuda"))

    num_classes_global = len(train_dataset.classes)
    print(f"Number of classes: {num_classes_global}")

    # Models
    generator = Generator(num_classes_global).to(run_device)
    discriminator = Discriminator(num_classes_global).to(run_device)

    # Loss & Optimizers
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # Tensorboard
    writer = SummaryWriter(log_dir="../runs/ConditionalGAN")

    print("Beginning Training (Conditional GAN)...")

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

            z = torch.randn(curr_batch_size, LATENT_DIM).to(run_device)
            fake_imgs = generator(z, labels)
            fake_output = discriminator(fake_imgs.detach(), labels)
            loss_d_fake = criterion(fake_output, fake_label)

            loss_d = loss_d_real + loss_d_fake
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
                fixed_noise = torch.randn(64, LATENT_DIM).to(run_device)
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
        test_noise = torch.randn(16, LATENT_DIM).to(run_device)
        test_labels = torch.randint(0, num_classes_global, (16,), device=run_device)
        test_imgs = generator(test_noise, test_labels)
        save_image(test_imgs, f"{SAVE_PATH}/final_samples.png", normalize=True)


if __name__ == "__main__":
    train()
