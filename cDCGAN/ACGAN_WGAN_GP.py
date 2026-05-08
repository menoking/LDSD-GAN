
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pytorch_fid.fid_score import calculate_fid_given_paths


# ============================================================
# Global Hyperparameters
# ============================================================
LATENT_DIM = 100
IMG_SIZE = 64
IMG_CHANNELS = 1
BATCH_SIZE = 64
LEARNING_RATE = 1e-4 # WGAN usually likes smaller LR
EPOCHS = 200         # WGAN converges slower
SAVE_PATH = "../Test_Results/Test_ACGAN_WGAN_GP_Results"

# WGAN-GP Specific
LAMBDA_GP = 10       # Gradient penalty lambda hyperparameter
N_CRITIC = 3         # Number of training steps for discriminator per iter (reduced from 5)
LAMBDA_AUX = 2.0     # Auxiliary classifier loss weight (increased from 1.0 to fix Class 1&4)
LABEL_SMOOTHING = 0.1  # Label smoothing for auxiliary loss

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
# Generator can keep BatchNorm (only Critic needs to avoid it)
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
# CRITICAL: No BatchNorm in Critic for WGAN-GP. Use InstanceNorm or LayerNorm.
class Discriminator(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        
        self.features = nn.Sequential(
            # Input: (1, 64, 64)
            nn.Conv2d(IMG_CHANNELS, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128, affine=True), # Changed from BN
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256, affine=True), # Changed from BN
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512, affine=True), # Changed from BN
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output layers
        # 1. Validity (Linear for Wasserstein)
        self.adv_layer = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )
        
        # 2. Classification (Softmax/CrossEntropy)
        self.aux_layer = nn.Sequential(
            nn.Conv2d(512, class_count, 4, 1, 0, bias=False),
        )

    def forward(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(image_tensor)
        
        validity = self.adv_layer(features).view(-1, 1)      # (N, 1)
        label_logits = self.aux_layer(features).view(features.size(0), -1) # (N, Class_Count)
        
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
    
    d_interpolates, _ = D(interpolates) # Ignore class output for GP
    
    fake = torch.ones(d_interpolates.shape, device=device, requires_grad=False)
    
    # Get gradient w.r.t. interpolates
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
    elif classname.find("BatchNorm") != -1: # InstanceNorm usually doesn't need this init but ok
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# ============================================================
# Visualization: FID & Loss Curves
# ============================================================
def plot_curves(history: dict, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    epochs = np.arange(1, len(history["d_losses"]) + 1)

    # Use a clean style
    plt.style.use("seaborn-v0_8-whitegrid")

    # --- Loss Curves ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. D Loss & G Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["d_losses"], label="D Loss", color="#e74c3c", linewidth=1.2)
    ax.plot(epochs, history["g_losses"], label="G Loss", color="#2ecc71", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Adversarial Loss (D & G)")
    ax.legend()

    # 2. Auxiliary Losses
    ax = axes[0, 1]
    ax.plot(epochs, history["d_aux_losses"], label="D Aux Loss", color="#e74c3c", linewidth=1.2)
    ax.plot(epochs, history["g_aux_losses"], label="G Aux Loss", color="#2ecc71", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Auxiliary Classification Loss")
    ax.legend()

    # 3. Gradient Penalty
    ax = axes[1, 0]
    ax.plot(epochs, history["gradient_penalties"], label="Gradient Penalty", color="#9b59b6", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Penalty")
    ax.set_title("Gradient Penalty")
    ax.legend()

    # 4. Classification Accuracy
    ax = axes[1, 1]
    ax.plot(epochs, history["acc_reals"], label="Real Acc", color="#3498db", linewidth=1.2)
    ax.plot(epochs, history["acc_fakes"], label="Fake Acc", color="#e67e22", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Discriminator Classification Accuracy")
    ax.legend()

    fig.suptitle("ACGAN-WGAN-GP Training Curves", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "loss_curves.svg"), format="svg", bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(save_dir, "loss_curves.png"), format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Plot] Loss curves saved to {save_dir}")

    # --- FID Curve ---
    if history["fid_scores"]:
        fig_fid, ax_fid = plt.subplots(figsize=(10, 6))
        fid_epochs = np.array(history["fid_epochs"])
        fid_scores = np.array(history["fid_scores"])
        ax_fid.plot(fid_epochs, fid_scores, marker="o", color="#e74c3c", linewidth=1.5, markersize=5, label="FID Score")

        # Mark best FID
        best_idx = np.argmin(fid_scores)
        ax_fid.scatter(fid_epochs[best_idx], fid_scores[best_idx], color="#2ecc71", s=120, zorder=5, label=f"Best FID = {fid_scores[best_idx]:.2f} @ Epoch {fid_epochs[best_idx]}")

        ax_fid.set_xlabel("Epoch", fontsize=12)
        ax_fid.set_ylabel("FID Score", fontsize=12)
        ax_fid.set_title("FID Score over Training", fontsize=14, fontweight="bold")
        ax_fid.legend(fontsize=11)
        ax_fid.grid(True, alpha=0.3)

        fig_fid.tight_layout()
        fig_fid.savefig(os.path.join(save_dir, "fid_curve.svg"), format="svg", bbox_inches="tight", dpi=300)
        fig_fid.savefig(os.path.join(save_dir, "fid_curve.png"), format="png", bbox_inches="tight", dpi=300)
        plt.close(fig_fid)
        print(f"[Plot] FID curve saved to {save_dir}")
    else:
        print("[Plot] No FID scores recorded, skipping FID curve.")


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

    # Optimizers (Adam with beta1=0 or 0.5 is common for WGAN-GP, usually beta1=0, beta2=0.9)
    optimizer_g = optim.Adam(gen_model.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    optimizer_d = optim.Adam(disc_model.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    
    # Tensorboard
    writer = SummaryWriter(log_dir="../runs/ACGAN_WGAN_GP")

    # Data collection for plotting
    history = {
        "d_losses": [],
        "g_losses": [],
        "d_aux_losses": [],
        "g_aux_losses": [],
        "gradient_penalties": [],
        "acc_reals": [],
        "acc_fakes": [],
        "fid_scores": [],
        "fid_epochs": [],
    }

    # Class loss with label smoothing
    auxiliary_loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    try:
        for epoch_index in range(EPOCHS):
            for i, (real_imgs, real_labels) in enumerate(train_loader):
                real_imgs = real_imgs.to(run_device)
                real_labels = real_labels.to(run_device)
                curr_batch_size = real_imgs.size(0)

                # ---------------------
                # Train Discriminator
                # ---------------------
                optimizer_d.zero_grad()

                # Real
                real_validity, real_aux_logits = disc_model(real_imgs)

                # Fake
                noise_vec = torch.randn(curr_batch_size, LATENT_DIM, device=run_device)
                gen_labels = torch.randint(0, num_classes_global, (curr_batch_size,), device=run_device)
                fake_imgs = gen_model(noise_vec, gen_labels)

                fake_validity, fake_aux_logits = disc_model(fake_imgs.detach())

                # WGAN Loss: -Mean(D(Real)) + Mean(D(Fake))
                loss_d_adv = -torch.mean(real_validity) + torch.mean(fake_validity)

                # Gradient Penalty
                gradient_penalty = compute_gradient_penalty(disc_model, real_imgs.data, fake_imgs.data, run_device)

                # Aux Loss (ACGAN)
                loss_d_aux = auxiliary_loss(real_aux_logits, real_labels) + \
                             auxiliary_loss(fake_aux_logits, gen_labels)

                # Calculate classification accuracy for monitoring
                with torch.no_grad():
                    pred_real = torch.argmax(real_aux_logits, dim=1)
                    pred_fake = torch.argmax(fake_aux_logits, dim=1)
                    acc_real = (pred_real == real_labels).float().mean()
                    acc_fake = (pred_fake == gen_labels).float().mean()

                # Total D Loss with weighted auxiliary loss
                loss_d = loss_d_adv + LAMBDA_GP * gradient_penalty + LAMBDA_AUX * loss_d_aux

                loss_d.backward()
                optimizer_d.step()

                # ---------------------
                # Train Generator (Every n_critic steps)
                # ---------------------
                if i % N_CRITIC == 0:
                    optimizer_g.zero_grad()

                    noise_vec = torch.randn(curr_batch_size, LATENT_DIM, device=run_device)
                    gen_labels = torch.randint(0, num_classes_global, (curr_batch_size,), device=run_device)
                    gen_imgs = gen_model(noise_vec, gen_labels)

                    fake_validity, fake_aux_logits = disc_model(gen_imgs)

                    # G Adversarial Loss: -Mean(D(Fake))
                    loss_g_adv = -torch.mean(fake_validity)

                    # G Aux Loss
                    loss_g_aux = auxiliary_loss(fake_aux_logits, gen_labels)

                    loss_g = loss_g_adv + LAMBDA_AUX * loss_g_aux

                    loss_g.backward()
                    optimizer_g.step()

            # Monitoring
            print(f"[Epoch {epoch_index + 1}/{EPOCHS}] D_Loss: {loss_d.item():.4f} | G_Loss: {loss_g.item():.4f} | Acc_Real: {acc_real.item():.3f} | Acc_Fake: {acc_fake.item():.3f}")
            writer.add_scalar("Loss/Discriminator", loss_d.item(), epoch_index + 1)
            writer.add_scalar("Loss/Generator", loss_g.item(), epoch_index + 1)
            writer.add_scalar("Loss/Gradient_Penalty", gradient_penalty.item(), epoch_index + 1)
            writer.add_scalar("Accuracy/Real", acc_real.item(), epoch_index + 1)
            writer.add_scalar("Accuracy/Fake", acc_fake.item(), epoch_index + 1)
            writer.add_scalar("Loss/D_Aux", loss_d_aux.item(), epoch_index + 1)
            if 'loss_g_aux' in locals():
                writer.add_scalar("Loss/G_Aux", loss_g_aux.item(), epoch_index + 1)

            # Collect data for plotting
            history["d_losses"].append(loss_d.item())
            history["g_losses"].append(loss_g.item())
            history["d_aux_losses"].append(loss_d_aux.item())
            history["gradient_penalties"].append(gradient_penalty.item())
            history["acc_reals"].append(acc_real.item())
            history["acc_fakes"].append(acc_fake.item())
            if 'loss_g_aux' in locals():
                history["g_aux_losses"].append(loss_g_aux.item())
            else:
                history["g_aux_losses"].append(0.0)

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
                writer.add_scalar("FID", fid_score, epoch_index + 1)
                history["fid_scores"].append(fid_score)
                history["fid_epochs"].append(epoch_index + 1)
                gen_model.train()

                # Real-time plot update
                plot_curves(history, SAVE_PATH)

        print("Training completed.")

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epoch_index + 1}. Generating curves with collected data...")
    except Exception as e:
        print(f"\nTraining error at epoch {epoch_index + 1}: {e}")
    finally:
        torch.save(gen_model.state_dict(), os.path.join(SAVE_PATH, "generator_final.pth"))
        torch.save(disc_model.state_dict(), os.path.join(SAVE_PATH, "discriminator_final.pth"))
        plot_curves(history, SAVE_PATH)
        writer.close()


if __name__ == "__main__":
    train()
