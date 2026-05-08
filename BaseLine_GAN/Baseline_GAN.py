import os
import random
import shutil
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from pytorch_fid.fid_score import calculate_fid_given_paths

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Global Hyperparameters
# ============================================================
LATENT_DIM = 100
IMG_SIZE = 64
IMG_CHANNELS = 1
IMAGE_FLAT_SIZE = IMG_CHANNELS * IMG_SIZE * IMG_SIZE
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
EPOCHS = 200
SAVE_PATH = "../Test_Results/Test_Gan_Results"

# Reproducibility
SEED = 42

# FID settings
FID_EVERY_N_EPOCHS = 20
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
# Generator (Unconditional MLP)
# ============================================================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(LATENT_DIM, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 2048),
            nn.Linear(2048, IMAGE_FLAT_SIZE),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), IMG_CHANNELS, IMG_SIZE, IMG_SIZE)


# ============================================================
# Discriminator (Unconditional MLP)
# ============================================================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(IMAGE_FLAT_SIZE, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


# ============================================================
# FID Calculation
# ============================================================
def calculate_fid_score(
    gen_model: nn.Module,
    train_loader: DataLoader,
    epoch_index: int,
    run_device: torch.device,
    writer: SummaryWriter,
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
                save_images_to_dir(real_imgs[:save_count], real_dir, saved_count)
                noise_vec = torch.randn(save_count, LATENT_DIM, device=run_device)
                fake_imgs = gen_model(noise_vec)
                save_images_to_dir(fake_imgs, fake_dir, saved_count)
                saved_count += save_count
        fid_score = calculate_fid_given_paths(
            [real_dir, fake_dir], batch_size=50, device=run_device, dims=2048, num_workers=0)
        print(f">>> Epoch {epoch_index} | FID = {fid_score:.4f}\n")
        writer.add_scalar("Metrics/FID", fid_score, epoch_index)
        return float(fid_score)
    except Exception as e:
        print(f"FID Error: {e}")
        return 999.0
    finally:
        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(fake_dir, ignore_errors=True)
        gen_model.train()


# ============================================================
# Visualization
# ============================================================
def plot_curves(history: dict, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    epochs = np.arange(1, len(history["d_losses"]) + 1)

    plt.style.use("seaborn-v0_8-whitegrid")

    # --- Loss Curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["d_losses"], label="D Loss", color="#e74c3c", linewidth=1.2)
    ax.plot(epochs, history["g_losses"], label="G Loss", color="#2ecc71", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Baseline MLP-GAN Training Loss")
    ax.legend()

    fig.suptitle("Baseline MLP-GAN Training Curves", fontsize=16, fontweight="bold", y=1.01)
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

        best_idx = np.argmin(fid_scores)
        ax_fid.scatter(fid_epochs[best_idx], fid_scores[best_idx], color="#2ecc71", s=120, zorder=5,
                       label=f"Best FID = {fid_scores[best_idx]:.2f} @ Epoch {fid_epochs[best_idx]}")

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


def train() -> None:
    if SEED is not None:
        set_seed(SEED)

    run_device = check_cuda_availability()
    os.makedirs(SAVE_PATH, exist_ok=True)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset_root = "../MSTAR/MSTAR_PUBLIC_MIXED_TARGETS_CD1/15_DEG/COL1/SCENE1"
    if not os.path.exists(dataset_root):
        dataset_root = "../MSTAR/PERSONAL_MSTAR/15_DEG"
        print(f"Warning: Original path not found, using {dataset_root}")

    train_dataset = ImageFolder(root=dataset_root, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(run_device.type == "cuda"))

    writer = SummaryWriter(log_dir="../runs/Baseline_GAN")

    # Models
    generator = Generator().to(run_device)
    discriminator = Discriminator().to(run_device)

    # Binary Cross Entropy Loss
    adversarial_loss = nn.BCELoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    print(f"Beginning Training (Baseline MLP-GAN)...")
    print(f"Dataset: {len(train_dataset)} samples, {len(train_dataset.classes)} classes")

    # Data collection for plotting
    history = {
        "d_losses": [],
        "g_losses": [],
        "fid_scores": [],
        "fid_epochs": [],
    }

    # Adversarial ground truth
    Tensor = torch.FloatTensor

    try:
        for epoch in range(EPOCHS):
            generator.train()
            discriminator.train()

            for i, (real_imgs, _) in enumerate(train_loader):
                curr_batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(run_device)

                # Ground truth labels
                valid = Tensor(curr_batch_size, 1).fill_(1.0).to(run_device)
                fake = Tensor(curr_batch_size, 1).fill_(0.0).to(run_device)

                # ---------------------
                # Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()

                # Real loss
                real_loss = adversarial_loss(discriminator(real_imgs), valid)

                # Fake loss
                z = torch.randn(curr_batch_size, LATENT_DIM).to(run_device)
                gen_imgs = generator(z)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

                # Total D loss
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                # ---------------------
                # Train Generator
                # ---------------------
                optimizer_G.zero_grad()

                # Generator wants D to think fakes are real
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)

                g_loss.backward()
                optimizer_G.step()

                if i % 50 == 0:
                    print(
                        f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(train_loader)}] "
                        f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                    )

            # Logging
            print(f"[Epoch {epoch + 1}/{EPOCHS}] D_Loss: {d_loss.item():.4f} | G_Loss: {g_loss.item():.4f}")
            writer.add_scalar("Loss/Discriminator", d_loss.item(), epoch + 1)
            writer.add_scalar("Loss/Generator", g_loss.item(), epoch + 1)

            history["d_losses"].append(d_loss.item())
            history["g_losses"].append(g_loss.item())

            # Save & FID
            if (epoch + 1) % FID_EVERY_N_EPOCHS == 0:
                with torch.no_grad():
                    fixed_noise = torch.randn(64, LATENT_DIM).to(run_device)
                    fake_samples = generator(fixed_noise)
                    save_image(fake_samples, f"{SAVE_PATH}/epoch_{epoch + 1}.png",
                               nrow=8, normalize=True)

                fid_score = calculate_fid_score(generator, train_loader, epoch + 1, run_device, writer)
                writer.add_scalar("FID", fid_score, epoch + 1)
                history["fid_scores"].append(fid_score)
                history["fid_epochs"].append(epoch + 1)

                plot_curves(history, SAVE_PATH)

        print("Training completed.")

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epoch + 1}. Generating curves with collected data...")
    except Exception as e:
        print(f"\nTraining error at epoch {epoch + 1}: {e}")
    finally:
        torch.save(generator.state_dict(), os.path.join(SAVE_PATH, "generator_final.pth"))
        torch.save(discriminator.state_dict(), os.path.join(SAVE_PATH, "discriminator_final.pth"))

        generator.eval()
        with torch.no_grad():
            test_noise = torch.randn(16, LATENT_DIM).to(run_device)
            test_imgs = generator(test_noise)
            save_image(test_imgs, f"{SAVE_PATH}/final_samples.png", normalize=True)

        plot_curves(history, SAVE_PATH)
        writer.close()


if __name__ == "__main__":
    train()
