
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
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder

from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import spectral_norm

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
EPOCHS = 200         # Increased to 200 to allow WGAN-GP convergence
SAVE_PATH = "../Test_Results/Test_ACGAN_WGAN_GP_Noise_Results"

# WGAN-GP Specific
LAMBDA_GP = 10
N_CRITIC = 2         # Keep at 2 for balance
LAMBDA_AUX_D = 1.0   # Reduced to 1.0 to balance realism and classification
LAMBDA_AUX_G = 1.0   # Reduced to 1.0 to balance realism and classification
LABEL_SMOOTHING = 0.05  # Reduced from 0.1 to allow stronger class signals


# Reproducibility
SEED = 42

# FID settings
FID_EVERY_N_EPOCHS = 10
FID_NUM_SAMPLES = 500

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

# ============================================================
# Custom Transforms
# ============================================================
class PadToSquare:
    def __call__(self, img):
        # img is a PIL Image
        w, h = img.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return TF.pad(img, padding, 0, 'constant')

# ============================================================
# Noise Injection Layer

# ============================================================
class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize with a very small value (0.001) to avoid disrupting generator learning initially
        # Allow it to grow if needed
        self.weight = nn.Parameter(torch.zeros(1) + 0.001)

    def forward(self, image):

        # image: [B, C, H, W]
        # noise: [B, 1, H, W] to be broadcasted
        batch, _, height, width = image.shape
        noise = torch.randn(batch, 1, height, width, device=image.device)
        return image + self.weight * noise


# ============================================================
# Conditional BatchNorm Layer (REMOVED)
# ============================================================
# class ConditionalBatchNorm2d(nn.Module):
#     ... (Removed for simplification)


# ============================================================
# Generator (with Noise Injection + CBN)
# ============================================================
class Generator(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        # Can reduce embedding back to class_count or keep it, 
        # but CBN does the heavy lifting now. Let's keep input simple.
        # We revert the input embedding to simple concatenation or just use LATENT_DIM
        self.label_embedding = nn.Embedding(class_count, class_count)

        # Initial Block
        # Input channels: LATENT_DIM + class_count
        self.fc = nn.ConvTranspose2d(LATENT_DIM + class_count, 512, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        # No noise injection at 4x4 usually, but can add if needed. optimize for structure first.
        
        # Block 2: 4x4 -> 8x8
        # Block 2: 4x4 -> 8x8
        self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.noise2 = NoiseInjection()
        self.bn2 = nn.BatchNorm2d(256)
        
        # Block 3: 8x8 -> 16x16
        self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.noise3 = NoiseInjection()
        self.bn3 = nn.BatchNorm2d(128)
        
        # Block 4: 16x16 -> 32x32
        self.conv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.noise4 = NoiseInjection()
        self.bn4 = nn.BatchNorm2d(64)
        
        # Block 5: 32x32 -> 64x64
        self.conv5 = nn.ConvTranspose2d(64, IMG_CHANNELS, 4, 2, 1, bias=False)
        # No BN/Noise at output usually
        
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, noise_vec: torch.Tensor, cond_labels: torch.Tensor) -> torch.Tensor:
        label_embed = self.label_embedding(cond_labels)
        combined = torch.cat([noise_vec, label_embed], dim=1)
        combined = combined.unsqueeze(2).unsqueeze(3)
        
        # 1. Initial 4x4
        x = self.fc(combined)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 2. 8x8
        x = self.conv2(x)
        x = self.noise2(x) # Inject Noise
        x = self.bn2(x)
        x = self.relu(x)
        
        # 3. 16x16
        x = self.conv3(x)
        x = self.noise3(x) # Inject Noise
        x = self.bn3(x)
        x = self.relu(x)
        
        # 4. 32x32
        x = self.conv4(x)
        x = self.noise4(x) # Inject Noise
        x = self.bn4(x)
        x = self.relu(x)
        
        # 5. Output 64x64
        x = self.conv5(x)
        x = self.tanh(x)
        
        return x


# ============================================================
# Discriminator (ACGAN - WGAN-GP Version)
# ============================================================
class Discriminator(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        
        self.features = nn.Sequential(
            # Input: (1, 64, 64)
            spectral_norm(nn.Conv2d(IMG_CHANNELS, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(128, affine=True), 
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
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
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
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
    real_dir = "./fid_real_noise"
    fake_dir = "./fid_fake_noise"
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
    except Exception as e:
        print(f"FID Calculation Failed: {e}")
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
    elif classname.find("BatchNorm") != -1 and classname != "ConditionalBatchNorm2d":
        # Initialize standard BN layers. 
        # Skip ConditionalBatchNorm2d wrapper (it handles its own init in __init__)
        # Also safely check for weight existence for internal layers
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

def train() -> None:
    if SEED is not None:
        set_seed(SEED)

    run_device = check_cuda_availability()
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    transform_pipeline = transforms.Compose([
        transforms.Grayscale(1),
        PadToSquare(),  # Pad to square to preserve aspect ratio
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    dataset_root = os.path.join("..", "MSTAR", "PERSONAL_MSTAR", "15_DEG")
    if not os.path.exists(dataset_root):
        print(f"Warning: Dataset not found at {dataset_root}")
        # Assuming user will fix path or provide it, but to prevent crash if running blindly:
        # dataset_root = "./dummy_data" 
    
    try:
        dataset = ImageFolder(root=dataset_root, transform=transform_pipeline)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(run_device.type=="cuda"))
        num_classes_global = len(dataset.classes)
        print(f"Number of classes: {num_classes_global}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Initialize Models
    gen_model = Generator(num_classes_global).to(run_device)
    disc_model = Discriminator(num_classes_global).to(run_device)
    
    gen_model.apply(weights_init_normal)
    disc_model.apply(weights_init_normal)

    optimizer_g = optim.Adam(gen_model.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    optimizer_d = optim.Adam(disc_model.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    
    writer = SummaryWriter(log_dir="../runs/ACGAN_WGAN_GP_Noise")
    auxiliary_loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    print("Beginning Training (ACGAN-WGAN-GP with Noise Injection)...")
    
    for epoch_index in range(EPOCHS):
        for i, (real_imgs, real_labels) in enumerate(train_loader):
            real_imgs = real_imgs.to(run_device)
            real_labels = real_labels.to(run_device)
            curr_batch_size = real_imgs.size(0)

            # NOTE: DiffAugment REMOVED - conflicts with NoiseInjection and ConditionalBatchNorm
            # The noise injection layer already provides stochasticity

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_d.zero_grad()

            # Train without DiffAugment - simpler and more stable with NoiseInjection
            real_validity, real_aux_logits = disc_model(real_imgs)
            
            # Calculate Real Accuracy
            pred_real = torch.argmax(real_aux_logits, dim=1)
            acc_real = (pred_real == real_labels).float().mean()
            
            noise_vec = torch.randn(curr_batch_size, LATENT_DIM, device=run_device)
            gen_labels = torch.randint(0, num_classes_global, (curr_batch_size,), device=run_device)
            fake_imgs = gen_model(noise_vec, gen_labels)
            
            fake_validity, fake_aux_logits = disc_model(fake_imgs.detach())
            
            # Calculate Fake Accuracy (Discriminator view)
            pred_fake = torch.argmax(fake_aux_logits, dim=1)
            acc_fake = (pred_fake == gen_labels).float().mean()

            # WGAN adversarial loss
            loss_d_adv = -torch.mean(real_validity) + torch.mean(fake_validity)
            
            # Gradient Penalty on clean images
            gradient_penalty = compute_gradient_penalty(disc_model, real_imgs.data, fake_imgs.detach().data, run_device)
            
            loss_d_aux = auxiliary_loss(real_aux_logits, real_labels) + auxiliary_loss(fake_aux_logits, gen_labels)
            
            # Total D Loss
            loss_d = loss_d_adv + LAMBDA_GP * gradient_penalty + LAMBDA_AUX_D * loss_d_aux
            
            loss_d.backward()
            optimizer_d.step()


            # ---------------------
            # Train Generator (Every n_critic steps)
            # ---------------------
            if i % N_CRITIC == 0:
                optimizer_g.zero_grad()
                
                # Regenerate fakes for G update
                noise_vec = torch.randn(curr_batch_size, LATENT_DIM, device=run_device)
                gen_labels = torch.randint(0, num_classes_global, (curr_batch_size,), device=run_device)
                gen_imgs = gen_model(noise_vec, gen_labels)
                
                # Train without DiffAugment
                gen_imgs_aug = gen_imgs  # No augmentation
                fake_validity, fake_aux_logits = disc_model(gen_imgs_aug)
                
                loss_g_adv = -torch.mean(fake_validity)

                loss_g_aux = auxiliary_loss(fake_aux_logits, gen_labels)
                
                # Total G Loss
                loss_g = loss_g_adv + LAMBDA_AUX_G * loss_g_aux
                
                loss_g.backward()
                optimizer_g.step()

        # Monitoring
        # Monitoring
        print(f"[Epoch {epoch_index + 1}/{EPOCHS}] D_Loss: {loss_d.item():.4f} | G_Loss: {loss_g.item():.4f} | Acc_R: {acc_real.item():.2f} | Acc_F: {acc_fake.item():.2f}")
        writer.add_scalar("Loss/Discriminator", loss_d.item(), epoch_index + 1)
        writer.add_scalar("Loss/Generator", loss_g.item(), epoch_index + 1)
        
        # Detailed Loss Components
        writer.add_scalar("Loss_Details/D_Adv", loss_d_adv.item(), epoch_index + 1)
        writer.add_scalar("Loss_Details/D_GP", gradient_penalty.item(), epoch_index + 1)
        writer.add_scalar("Loss_Details/D_Aux", loss_d_aux.item(), epoch_index + 1)
        
        # Accuracy Logging
        writer.add_scalar("Accuracy/Real", acc_real.item(), epoch_index + 1)
        writer.add_scalar("Accuracy/Fake", acc_fake.item(), epoch_index + 1)

        if 'loss_g_adv' in locals():
            writer.add_scalar("Loss_Details/G_Adv", loss_g_adv.item(), epoch_index + 1)
            writer.add_scalar("Loss_Details/G_Aux", loss_g_aux.item(), epoch_index + 1)


        # Save & FID
        if (epoch_index + 1) % FID_EVERY_N_EPOCHS == 0:
            gen_model.eval()
            with torch.no_grad():
                noise_vec = torch.randn(64, LATENT_DIM, device=run_device)
                sample_labels = torch.arange(num_classes_global, device=run_device).repeat(20)[:64]
                samples = gen_model(noise_vec, sample_labels)
            
            save_image(
                samples.detach().cpu(),
                os.path.join(SAVE_PATH, f"epoch_{epoch_index + 1}.png"),
                nrow=8,
                normalize=True,
            )
            
            fid_score = calculate_fid_score(gen_model, train_loader, epoch_index + 1, run_device, num_classes_global)
            writer.add_scalar("Metrics/FID", fid_score, epoch_index + 1)
            gen_model.train()

    print("Training completed.")
    torch.save(gen_model.state_dict(), os.path.join(SAVE_PATH, "generator_final.pth"))
    torch.save(disc_model.state_dict(), os.path.join(SAVE_PATH, "discriminator_final.pth"))
    writer.close()


if __name__ == "__main__":
    train()
