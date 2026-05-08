
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import importlib.util

# ============================================================
# Project Path Setup
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MSTAR_PATH = os.path.join(PROJECT_ROOT, "MSTAR", "PERSONAL_MSTAR", "15_DEG")

# ============================================================
# Configuration
# ============================================================
LATENT_DIM_DEFAULT = 100
IMG_SIZE_DEFAULT = 64
NUM_SAMPLES = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Configuration
# Name: (Directory, FileName, WeightsPath, is_conditional)
# Note: Weight paths are relative to PROJECT_ROOT
MODELS_CONFIG = {
    "MLP_GAN": {
        "dir": "BaseLine_GAN",
        "file": "Baseline_GAN.py",
        "weights": "Test_Results/Test_Gan_Results/generator_final.pth",
        "conditional": False
    },
    "CGAN": {
        "dir": "ConditionalGAN",
        "file": "ConditionalGAN.py",
        "weights": "Test_Results/Test_cGAN_Results/generator_final.pth",
        "conditional": True
    },
    "cDCGAN": {
        "dir": "cDCGAN",
        "file": "cDCGAN.py",
        "weights": "Test_Results/Test_cDCGAN_Results/generator_final.pth",
        "conditional": True
    },
    "ACGAN": {
        "dir": "cDCGAN",
        "file": "ACGAN.py",
        "weights": "Test_Results/Test_ACGAN_Results/generator_final.pth",
        "conditional": True
    },
    "AC_WGAN_GP": {
        "dir": "cDCGAN",
        "file": "ACGAN_WGAN_GP.py",
        "weights": "Test_Results/Test_ACGAN_WGAN_GP_Results/generator_final.pth",
        "conditional": True
    },
    "LDSD_GAN": {
        "dir": "cDCGAN",
        "file": "ACGAN_WGAN_GP_Physics.py",
        "weights": "Test_Results/Test_ACGAN_WGAN_GP_Physics_Results/generator_final.pth",
        "conditional": True
    },
    # The following are evaluated for the table but excluded from the main plot
    "cDCGAN_Physics": {
        "dir": "cDCGAN",
        "file": "cDCGAN_Physics.py",
        "weights": "Test_Results/Test_cDCGAN_Physics_Results/generator_final.pth",
        "conditional": True,
        "no_plot": True
    },
    "Ablation_A_Base": {
        "dir": "Ablation",
        "file": "ablation_experiment.py",
        "weights": "Test_Results/A_Baseline/generator_final.pth",
        "conditional": True,
        "config": "A",
        "no_plot": True
    },
    "Ablation_B_Log": {
        "dir": "Ablation",
        "file": "ablation_experiment.py",
        "weights": "Test_Results/B_LogOnly/generator_final.pth",
        "conditional": True,
        "config": "B",
        "no_plot": True
    },
    "Ablation_C_Phys": {
        "dir": "Ablation",
        "file": "ablation_experiment.py",
        "weights": "Test_Results/C_FullPhysics/generator_final.pth",
        "conditional": True,
        "config": "C",
        "no_plot": True
    }
}

# ============================================================
# Helper Functions
# ============================================================

def get_generator_class(model_name, cfg):
    """
    Dynamically loads the Generator class from the specified file.
    Follows the pattern in comparative_experiment.py.
    """
    model_dir = os.path.join(PROJECT_ROOT, cfg["dir"])
    file_path = os.path.join(model_dir, cfg["file"])
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Generator file not found at: {file_path}")

    # Temporary add model directory to sys.path to handle internal dependencies (e.g. DiffAugment_pytorch)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    
    # Also add project root so modules can find sibling folders
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    # Also add cDCGAN specifically as many things depend on it
    cdc_dir = os.path.join(PROJECT_ROOT, "cDCGAN")
    if cdc_dir not in sys.path:
        sys.path.insert(0, cdc_dir)

    spec = importlib.util.spec_from_file_location(model_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[model_name] = module
    spec.loader.exec_module(module)
    
    # Extract constants if available, else use defaults
    latent_dim = getattr(module, 'LATENT_DIM', LATENT_DIM_DEFAULT)
    img_size = getattr(module, 'IMG_SIZE', IMG_SIZE_DEFAULT)
    
    return module.Generator, latent_dim, img_size

def get_enl(pixels):
    """Calculate Equivalent Number of Looks (ENL)."""
    intensity = pixels ** 2
    mean_val = np.mean(intensity)
    var_val = np.var(intensity)
    return (mean_val ** 2) / var_val if var_val > 0 else 0

def inverse_map_pixels(img_tensor):
    """Map [-1, 1] tensor to [0, 255] numpy array (Amplitude)."""
    img = img_tensor.cpu().detach().numpy()
    img = (img + 1) / 2.0 * 255.0
    return img

def load_real_samples(root, num_samples, img_size):
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.ImageFolder(root=root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    real_imgs, _ = next(iter(dataloader))
    return real_imgs, len(dataset.classes)

# ============================================================
# Main Analysis
# ============================================================
def main():
    print("Starting Physics Distribution Test (Final Robust Version)...")
    
    if not os.path.exists(MSTAR_PATH):
        print(f"Error: MSTAR path not found: {MSTAR_PATH}")
        return

    # 1. Load Real Data Distribution
    print("Loading Real MSTAR samples...")
    real_imgs_tensor, num_classes = load_real_samples(MSTAR_PATH, NUM_SAMPLES, IMG_SIZE_DEFAULT)
    real_data = inverse_map_pixels(real_imgs_tensor).flatten()
    
    # Calculate Real Stats
    real_enl = get_enl(real_data)
    r_gamma_params = stats.gamma.fit(real_data, loc=0)
    r_rayleigh_params = stats.rayleigh.fit(real_data, loc=0)
    ks_gamma_real, _ = stats.kstest(real_data, 'gamma', args=r_gamma_params)
    ks_rayleigh_real, _ = stats.kstest(real_data, 'rayleigh', args=r_rayleigh_params)
    
    results = {
        "Real_MSTAR": {
            "ENL": real_enl,
            "KS_G": ks_gamma_real,
            "KS_R": ks_rayleigh_real,
            "KL": 0.0
        }
    }
    
    # Plot Setup
    fig, axes = plt.subplots(2, 4, figsize=(24, 18))
    axes = axes.flatten()
    
    # 2. Evaluate Models
    plot_idx = 0
    for model_name, cfg in MODELS_CONFIG.items():
        is_plotted = not cfg.get("no_plot", False)
        if is_plotted:
            ax = axes[plot_idx]
            plot_idx += 1
        
        print(f"\nEvaluating {model_name}...")
        
        try:
            # Dynamic Load
            GeneratorClass, l_dim, i_size = get_generator_class(model_name, cfg)
            
            # Instantiate
            if "config" in cfg:
                gen = GeneratorClass(num_classes, cfg["config"]).to(DEVICE)
            else:
                gen = GeneratorClass(num_classes).to(DEVICE) if cfg["conditional"] else GeneratorClass().to(DEVICE)
            
            # Load Weights
            weights_path = os.path.join(PROJECT_ROOT, cfg["weights"])
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=DEVICE)
                # Cleanup state_dict if it has 'module.' prefix
                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
                gen.load_state_dict(state_dict)
                print(f"  Loaded weights from: {cfg['weights']}")
            else:
                print(f"  Warning: Weights not found at {weights_path}. Using random initialization.")
            
            gen.eval()
            
            # Generate Samples
            with torch.no_grad():
                z = torch.randn(NUM_SAMPLES, l_dim, device=DEVICE)
                if cfg["conditional"]:
                    labels = torch.randint(0, num_classes, (NUM_SAMPLES,), device=DEVICE)
                    fake_imgs = gen(z, labels)
                else:
                    fake_imgs = gen(z)
            
            fake_data = inverse_map_pixels(fake_imgs).flatten()
            
            # Metrics
            enl_val = get_enl(fake_data)
            gamma_params = stats.gamma.fit(fake_data, loc=0) 
            rayleigh_params = stats.rayleigh.fit(fake_data, loc=0)
            
            ks_gamma, _ = stats.kstest(fake_data, 'gamma', args=gamma_params)
            ks_rayleigh, _ = stats.kstest(fake_data, 'rayleigh', args=rayleigh_params)
            
            # KL Divergence
            hist_real, _ = np.histogram(real_data, bins=50, range=(0, 255), density=True)
            hist_fake, _ = np.histogram(fake_data, bins=50, range=(0, 255), density=True)
            kl_div = stats.entropy(hist_real + 1e-10, hist_fake + 1e-10)
            
            results[model_name] = {"ENL": enl_val, "KS_G": ks_gamma, "KS_R": ks_rayleigh, "KL": kl_div}
            
            if is_plotted:
                # Subplot
                sns.histplot(fake_data, bins=50, stat="density", color="skyblue", alpha=0.6, ax=ax, label='Generated')
                x = np.linspace(0, 255, 200)
                ax.plot(x, stats.gamma.pdf(x, *gamma_params), 'r-', label=f'Gamma (KS={ks_gamma:.2f})')
                ax.plot(x, stats.rayleigh.pdf(x, *rayleigh_params), 'g--', label=f'Rayleigh (KS={ks_rayleigh:.2f})')
                ax.set_title(f"{model_name}\nENL={enl_val:.1f} | KL={kl_div:.2f}", fontsize=18)
                ax.legend(fontsize=16)
                ax.set_xlim(0, 255)

        except Exception as e:
            print(f"  Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            if is_plotted:
                ax.text(0.5, 0.5, f"Error:\n{model_name}", ha='center', va='center', color='red')
                ax.set_title(f"Error: {model_name}")

    # 3. Real Reference Plot
    ax_ref = axes[6] 
    # Hide indices 7 (last one in 2x4)
    if len(axes) > 7:
        axes[7].axis('off')
    sns.histplot(real_data, bins=50, stat="density", color="gray", alpha=0.3, ax=ax_ref, label='Real MSTAR')
    x = np.linspace(0, 255, 200)
    ax_ref.plot(x, stats.gamma.pdf(x, *r_gamma_params), 'r-', label=f'Gamma (KS={ks_gamma_real:.2f})')
    ax_ref.plot(x, stats.rayleigh.pdf(x, *r_rayleigh_params), 'g--', label=f'Rayleigh (KS={ks_rayleigh_real:.2f})')
    ax_ref.set_title(f"Real MSTAR Reference\nENL={real_enl:.1f}", fontsize=18)
    ax_ref.legend(fontsize=16)
    ax_ref.set_xlim(0, 255)
    
    plt.tight_layout()
    output_img = "Physics_Distribution_Comparison.tiff"
    plt.savefig(output_img, dpi=300, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    print(f"\nFinal visualization saved to {output_img} (High-resolution TIFF)")
    
    # Try to show the plot window
    try:
        plt.show()
    except Exception:
        print("Note: Could not open GUI window. Check the saved .png file.")
    
    # 4. Final Summary Table
    print("\n" + "="*85)
    print(f"{'Model':<25} | {'ENL':<8} | {'KL Div':<8} | {'KS Gamma':<10} | {'KS Rayleigh':<10}")
    print("-" * 85)
    for name, res in results.items():
        print(f"{name:<25} | {res['ENL']:<8.2f} | {res['KL']:<8.4f} | {res['KS_G']:<10.4f} | {res['KS_R']:<10.4f}")
    print("="*85)

if __name__ == "__main__":
    main()
