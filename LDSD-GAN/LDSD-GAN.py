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
from torch.utils.tensorboard import SummaryWriter

from pytorch_fid.fid_score import calculate_fid_given_paths
from DiffAugment_pytorch import DiffAugment

# ============================================================
# Global Hyperparameters
# ============================================================
LATENT_DIM = 100  # 初始噪声维度
IMG_SIZE = 64  # 预处理成图像尺寸
IMG_CHANNELS = 1  # 图像通道
BATCH_SIZE = 64  # 批数量
EPOCHS = 200  # 训练轮次
# TTUR (Two-Time-Scale Update Rule)
LEARNING_RATE_G = 2e-4  # 生成器学习率
LEARNING_RATE_D = 2e-4  # 判别器学习率
SAVE_PATH = "../Test_Results/Test_ACGAN_WGAN_GP_Physics_Results"  # 模型及生成图像存储路径

# WGAN-GP Specific
LAMBDA_GP = 10  # 梯度惩罚系数，约束判别器梯度模长
N_CRITIC = 3  # 训练比例：判别器/生成器
LAMBDA_AUX = 1.0  # 辅助分类损失权重。4.0 -> 1.0
LABEL_SMOOTHING = 0.1  # 标签平滑，类似于死区控制

# DiffAugment Policy
DIFFAUG_POLICY = 'color,translation'  # 数据增强策略

# Physics Parameters
# We use Log-Normal approximation: Log(Image) = Log(Reflectance) + Noise
# Initial Sigma corresponds to approximate Speckle intensity
INITIAL_SIGMA = 0.05  # 初始噪声强度
LAMBDA_PHYS = 10.0  # 统计一致性损失权重

# FID settings
FID_EVERY_N_EPOCHS = 10  # FID测试频率，间隔10epochs
FID_NUM_SAMPLES = 500  # FID 采样数量

# Reproducibility
SEED = 42  # 随机数种子


# ============================================================
# Utilities
# ============================================================
# 随机数设置函数
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 强制清空目录并创建新文件夹
def ensure_empty_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

# 保存图像至指定目录
def save_images_to_dir(images: torch.Tensor, directory: str, start_index: int = 0) -> None:
    for idx, img in enumerate(images):  # enumerate用来列表、元组或字符串组合为一个索引序列
        save_image(
            img.detach().cpu(),  # 分离计算图
            os.path.join(directory, f"{start_index + idx}.png"),
            normalize=True,
        )


# ============================================================
# Learnable Log-Physics Layer
# ============================================================
# 物理层
class LogPhysicsLayer(nn.Module):
    """
    Simulates SAR Speckle Noise in Log-Domain.
    Log(I) = Log(R) + N
    Where N ~ N(0, sigma^2)
    This is equivalent to multiplicative noise in linear domain, 
    but numerically much more stable for GANs.
    NOW: Class-Conditional Sigma!
    """

    def __init__(self, num_classes, init_sigma=INITIAL_SIGMA):
        super(LogPhysicsLayer, self).__init__()
        # Learnable noise intensity PER CLASS
        # Shape: [num_classes, 1, 1, 1]
        val = np.log(init_sigma)
        # Initialize all classes with init_sigma
        self.log_sigma = nn.Parameter(torch.full((num_classes, 1, 1, 1), val))

    def forward(self, log_reflectance: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_reflectance: Estimated log-reflectance from generator
            labels: Class labels for the batch
        Returns:
            Noisy log-image (or mapped to [-1, 1])
        """
        # Select sigma for each sample based on class
        # batch_log_sigma: [Batch, 1, 1, 1]
        batch_log_sigma = self.log_sigma[labels]
        sigma = torch.exp(batch_log_sigma)

        if self.training:
            # Additive Gaussian Noise (Log-Speckle)
            noise = torch.randn_like(log_reflectance) * sigma
            noisy_out = log_reflectance + noise
        else:
            # During evaluation, we still add noise to simulate SAR texture, 
            # OR we can reduce it. For physics fidelity, we keep it.
            noise = torch.randn_like(log_reflectance) * sigma
            noisy_out = log_reflectance + noise

        return noisy_out


# ============================================================
# Generator (ACGAN + Log-Physics)
# ============================================================
class Generator(nn.Module):
    def __init__(self, class_count: int):
        super(Generator, self).__init__()
        # 嵌入层函数：词表大小，嵌入维度
        self.label_embedding = nn.Embedding(class_count, class_count)

        self.net = nn.Sequential(
            # Input: LATENT_DIM + class_count -> 4x4
            nn.ConvTranspose2d(LATENT_DIM + class_count, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),  # 批归一化，参数由下面的函数weights_init_normal决定
            nn.ReLU(True),

            # 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64x64 Output
            nn.ConvTranspose2d(64, IMG_CHANNELS, 4, 2, 1, bias=False),
            # Note: No Tanh here yet! We output "Log-Reflectance" features
        )

        self.physics_layer = LogPhysicsLayer(num_classes=class_count)
        self.final_act = nn.Tanh()

    def forward(self, noise_vec: torch.Tensor, cond_labels: torch.Tensor) -> torch.Tensor:
        label_embed = self.label_embedding(cond_labels)
        combined = torch.cat([noise_vec, label_embed], dim=1)  # 拼接向量
        combined = combined.unsqueeze(2).unsqueeze(3)  # 升维：二维原张量 -> 四维张量

        # 1. Generate Log-Reflectance Features
        log_reflectance = self.net(combined)  # 生成器卷积网络

        # 2. Add Physics Noise (Additive in Log-Domain)
        noisy_log = self.physics_layer(log_reflectance, cond_labels)  # 通过标签添加添加对应物理噪声

        # 3. Map to Image Domain [-1, 1]
        # This assumes the dataset is normalized to [-1, 1]
        out = self.final_act(noisy_log)  # tan归一化

        return out


# ============================================================
# Discriminator (ACGAN - WGAN-GP Version)
# ============================================================
class Discriminator(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()

        # Reference cDCGAN: Embed labels as an image channel
        self.label_embedding = nn.Embedding(class_count, IMG_SIZE * IMG_SIZE)

        self.features = nn.Sequential(
            # Input: (1 real/fake image + 1 label channel) = 2 channels
            nn.Conv2d(IMG_CHANNELS + 1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128, affine=True),  # 参数由下面的函数weights_init_normal决定
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output: Single validity scalar for WGAN
        self.adv_layer = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, image_tensor: torch.Tensor, cond_labels: torch.Tensor) -> torch.Tensor:
        # Construct label image channel
        label_embed = self.label_embedding(cond_labels)
        label_img = label_embed.view(-1, 1, IMG_SIZE, IMG_SIZE)

        # Concatenate
        combined = torch.cat([image_tensor, label_img], dim=1)

        features = self.features(combined)
        validity = self.adv_layer(features).view(-1, 1)
        return validity


# ============================================================
# Gradient Penalty
# ============================================================
# 计算GP惩罚项
def compute_gradient_penalty(D, real_samples, fake_samples, labels, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)   # 0~1随机图（介于真假之间）
    d_interpolates = D(interpolates, labels)  # 判别随机图并得分
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
# 计算FID数值
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
# 获取计算设备
def check_cuda_availability():
    if torch.cuda.is_available():
        print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("CUDA is NOT available. Using CPU. Warning: Training will be extremely slow.")
        return torch.device("cpu")

# 归一化参数初始化
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train() -> None:
    # 设置随机数种子
    if SEED is not None:
        set_seed(SEED)

    # 设置计算设备
    run_device = check_cuda_availability()
    # 新建存储路径
    os.makedirs(SAVE_PATH, exist_ok=True)

    # 数据预处理
    transform_pipeline = transforms.Compose([
        transforms.Grayscale(1),   # 灰度单通道
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 尺寸缩放
        transforms.ToTensor(),  # 张量化
        transforms.Normalize((0.5,), (0.5,))  # 正则化 [0,1] -> [-1,+1]
    ])

    # 数据集根目录
    dataset_root = os.path.join("..", "MSTAR", "PERSONAL_MSTAR", "15_DEG")
    # 不存在则终止
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset not found at {dataset_root}")
        return

    # 加载数据集
    dataset = ImageFolder(root=dataset_root, transform=transform_pipeline)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                              pin_memory=(run_device.type == "cuda"))
    # 获取类别数
    num_classes_global = len(dataset.classes)
    print(f"Number of classes: {num_classes_global}")

    # 实例化生成器判别器
    gen_model = Generator(num_classes_global).to(run_device)
    disc_model = Discriminator(num_classes_global).to(run_device)

    # 设置权值
    gen_model.apply(weights_init_normal)
    disc_model.apply(weights_init_normal)

    # 优化器实例化
    optimizer_g = optim.Adam(gen_model.parameters(), lr=LEARNING_RATE_G, betas=(0.0, 0.9))
    optimizer_d = optim.Adam(disc_model.parameters(), lr=LEARNING_RATE_D, betas=(0.0, 0.9))

    # Tensor Board日志对象
    writer = SummaryWriter(log_dir="../runs/ACGAN_WGAN_GP_Physics")

    # Label Smoothing (No Manual Class Weights)
    # 类别辅助判别器损失
    auxiliary_loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    print(f"Beginning Training (ACGAN-WGAN-GP-Physics + DiffAugment)...")
    print(f"DiffAugment Policy: {DIFFAUG_POLICY}")
    print(f"Initial Sigma: {INITIAL_SIGMA}")

    # 训练
    for epoch_index in range(EPOCHS):
        for i, (real_imgs, real_labels) in enumerate(train_loader):  # 取元素
            real_imgs = real_imgs.to(run_device)
            real_labels = real_labels.to(run_device)
            # 当前批次数量，存在于第一维度
            curr_batch_size = real_imgs.size(0)

            # ---------------------
            # Train Discriminator
            # ---------------------
            # 优化器梯度清零
            optimizer_d.zero_grad()

            # 生成噪声向量
            noise_vec = torch.randn(curr_batch_size, LATENT_DIM, device=run_device)
            # 分配一个批次里每类别数量
            samples_per_class = curr_batch_size // num_classes_global  # //为向下取整除法

            # 填充对应类别索引，形成批次标签
            gen_labels = torch.cat([
                torch.full((samples_per_class,), idx, dtype=torch.long, device=run_device)
                for idx in range(num_classes_global)
            ])
            # 随机补充批次中不满的标签
            if len(gen_labels) < curr_batch_size:
                rem = curr_batch_size - len(gen_labels)
                gen_labels = torch.cat([gen_labels, torch.randint(0, num_classes_global, (rem,), device=run_device)])
            gen_labels = gen_labels[:curr_batch_size]  # 裁剪标签向量，确保长度为curr_batch_size

            # 生成虚假样本
            fake_imgs = gen_model(noise_vec, gen_labels)

            # 对真实图片与假图片都做微分增强
            real_imgs_aug = DiffAugment(real_imgs, policy=DIFFAUG_POLICY)
            fake_imgs_aug = DiffAugment(fake_imgs.detach(), policy=DIFFAUG_POLICY)

            # 判别器分数
            real_validity = disc_model(real_imgs_aug, real_labels)
            fake_validity = disc_model(fake_imgs_aug, gen_labels)

            # 判别器损失函数 WGAN Loss
            loss_d_adv = -torch.mean(real_validity) + torch.mean(fake_validity)
            # 计算梯度惩罚 Gradient Penalty
            gradient_penalty = compute_gradient_penalty(disc_model, real_imgs_aug, fake_imgs_aug, real_labels,
                                                        run_device)
            # WGAN判别器总体损失函数 Total D Loss
            loss_d = loss_d_adv + LAMBDA_GP * gradient_penalty

            # 反向传播与参数更新
            loss_d.backward()
            optimizer_d.step()

            # ---------------------
            # Train Generator (Every n_critic steps)
            # ---------------------
            if i % N_CRITIC == 0:  # 控制训练比例
                optimizer_g.zero_grad()

                # Resample balanced labels for G update
                noise_vec = torch.randn(curr_batch_size, LATENT_DIM, device=run_device)
                gen_labels = torch.cat([
                    torch.full((samples_per_class,), idx, dtype=torch.long, device=run_device)
                    for idx in range(num_classes_global)
                ])
                if len(gen_labels) < curr_batch_size:
                    rem = curr_batch_size - len(gen_labels)
                    gen_labels = torch.cat(
                        [gen_labels, torch.randint(0, num_classes_global, (rem,), device=run_device)])
                gen_labels = gen_labels[:curr_batch_size]

                gen_imgs = gen_model(noise_vec, gen_labels)

                # 微分增强与图像判别
                gen_imgs_aug = DiffAugment(gen_imgs, policy=DIFFAUG_POLICY)
                fake_validity = disc_model(gen_imgs_aug, gen_labels)
                loss_g_adv = -torch.mean(fake_validity)

                # 统计一致性损失，均值和标准差对齐
                mean_real, std_real = real_imgs.mean(), real_imgs.std()
                mean_fake, std_fake = gen_imgs.mean(), gen_imgs.std()
                loss_stat = torch.abs(mean_fake - mean_real) + torch.abs(std_fake - std_real)

                # 生成器损失函数
                loss_g = loss_g_adv + LAMBDA_PHYS * loss_stat

                loss_g.backward()
                optimizer_g.step()

        # Monitoring
        with torch.no_grad():
            sigmas = torch.exp(gen_model.physics_layer.log_sigma).squeeze()
            sigma_c1 = sigmas[1].item() if sigmas.numel() > 1 else sigmas.item()
            sigma_mean = sigmas.mean().item()

        print(
            f"[Epoch {epoch_index + 1}/{EPOCHS}] D_Loss: {loss_d.item():.4f} | G_Loss: {loss_g.item():.4f} | Stat_Loss: {loss_stat.item():.4f} | Sig_Mean: {sigma_mean:.3f}")
        writer.add_scalar("Loss/Discriminator", loss_d.item(), epoch_index + 1)
        writer.add_scalar("Loss/Generator", loss_g.item(), epoch_index + 1)
        writer.add_scalar("Loss/Statistical", loss_stat.item(), epoch_index + 1)
        writer.add_scalar("Physics/Sigma_Mean", sigma_mean, epoch_index + 1)
        writer.add_scalar("Physics/Sigma_Class1", sigma_c1, epoch_index + 1)

        # Save & FID
        if (epoch_index + 1) % FID_EVERY_N_EPOCHS == 0:
            gen_model.eval()  # 评估模式：在不更新模型参数（一般末尾应用阶段调用）
            with torch.no_grad():
                noise_vec = torch.randn(64, LATENT_DIM, device=run_device)
                # 生成一维等差数列标签张量 ,eg.[0,1,2,3,4], 重复20次拼接后截取
                sample_labels = torch.arange(num_classes_global, device=run_device).repeat(100)[:64]
                samples = gen_model(noise_vec, sample_labels)

            save_image(
                samples.detach().cpu(),
                os.path.join(SAVE_PATH, f"epoch_{epoch_index + 1}.png"),
                nrow=8,
                normalize=True,
            )

            fid_score = calculate_fid_score(gen_model, train_loader, epoch_index + 1, run_device, num_classes_global)
            writer.add_scalar("Metrics/FID", fid_score, epoch_index + 1)
            gen_model.train()  # 切换为训练模式

    print("Training completed.")
    # 保存模型文件
    torch.save(gen_model.state_dict(), os.path.join(SAVE_PATH, "generator_final.pth"))
    torch.save(disc_model.state_dict(), os.path.join(SAVE_PATH, "discriminator_final.pth"))
    writer.close()


if __name__ == "__main__":
    train()
