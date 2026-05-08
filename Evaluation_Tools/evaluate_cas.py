import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms

# Add cDCGAN directory to path for importing Generator
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cDCGAN"))

try:
    from ACGAN import Generator, LATENT_DIM, IMG_SIZE
except ImportError:
    # Fallback or error handling
    print("Error: Could not import Generator from ACGAN.py. Make sure the file exists.")
    sys.exit(1)

# Configuration
CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "classifier_resnet18_best.pth")
GENERATOR_PATH = "TODO_provide_generator_checkpoint_path.pth"  # Placeholder
NUM_CLASSES = 6  # Set to 6 to match MSTAR/PERSONAL_MSTAR/15_DEG
SAMPLES_PER_CLASS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_TRIALS = 5  # Number of trials for more stable evaluation


def load_classifier(path, num_classes):
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"Loaded classifier from {path}")
    else:
        print(f"Warning: Classifier checkpoint not found at {path}")

    model.to(DEVICE)
    model.eval()
    return model


def load_generator(path, num_classes):
    gen = Generator(num_classes)
    if os.path.exists(path):
        # Handle potential key prefixes if saved differently
        state_dict = torch.load(path, map_location=DEVICE)
        gen.load_state_dict(state_dict, strict=False)
        print(f"Loaded generator from {path}")
    else:
        print(f"Warning: Generator checkpoint not found at {path}. Using random weights.")

    gen.to(DEVICE)
    gen.eval()
    return gen


def main():
    if len(sys.argv) > 1:
        gen_path = sys.argv[1]
    else:
        gen_path = GENERATOR_PATH

    print(f"Evaluating Generator: {gen_path}")

    # Load Models
    classifier = load_classifier(CLASSIFIER_PATH, NUM_CLASSES)
    generator = load_generator(gen_path, NUM_CLASSES)

    # Store results from multiple trials
    all_trials_class_acc = []
    all_trials_overall_acc = []

    for trial in range(NUM_TRIALS):
        print(f"\n=== Trial {trial + 1}/{NUM_TRIALS} ===")
        
        total_correct = 0
        total_samples = 0
        class_accuracies = {}

        with torch.no_grad():
            for class_idx in range(NUM_CLASSES):
                # Generate fake samples for this class
                noise = torch.randn(SAMPLES_PER_CLASS, LATENT_DIM, device=DEVICE)
                labels = torch.full((SAMPLES_PER_CLASS,), class_idx, dtype=torch.long, device=DEVICE)

                fake_imgs = generator(noise, labels)

                # Classify
                outputs = classifier(fake_imgs)
                _, preds = torch.max(outputs, 1)

                correct = (preds == labels).sum().item()
                acc = correct / SAMPLES_PER_CLASS
                class_accuracies[class_idx] = acc

                total_correct += correct
                total_samples += SAMPLES_PER_CLASS

                print(f"Class {class_idx}: Acc = {acc:.2f}")

        overall_acc = total_correct / total_samples
        all_trials_class_acc.append(class_accuracies)
        all_trials_overall_acc.append(overall_acc)
        print(f"Trial {trial + 1} Overall CAS: {overall_acc:.4f}")

    # Calculate statistics across trials
    print("\n" + "="*50)
    print("FINAL RESULTS (averaged over {} trials)".format(NUM_TRIALS))
    print("="*50)
    
    # Per-class statistics
    import numpy as np
    for class_idx in range(NUM_CLASSES):
        class_accs = [trial[class_idx] for trial in all_trials_class_acc]
        mean_acc = np.mean(class_accs)
        std_acc = np.std(class_accs)
        print(f"Class {class_idx}: Mean Acc = {mean_acc:.3f} ± {std_acc:.3f}")
    
    # Overall statistics
    mean_overall = np.mean(all_trials_overall_acc)
    std_overall = np.std(all_trials_overall_acc)
    print("-" * 50)
    print(f"Overall CAS: {mean_overall:.4f} ± {std_overall:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
