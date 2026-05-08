
import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset, TensorDataset, Subset
from torchvision import transforms, datasets, models
import torchvision.transforms.functional as TF
import numpy as np
import time
import importlib.util

# ============================================================
# 用户配置区域 (User Configuration)
# ============================================================
# 动态获取项目根目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 真实数据路径
REAL_DATA_DIR = os.path.join(PROJECT_ROOT, "MSTAR", "PERSONAL_MSTAR", "45_DEG")

# GAN 生成器权重路径
# 根据 GENERATOR_TYPE 选择对应的 checkpoint:
# - 'Baseline_GAN'              -> "Test_Gan_Results/generator_final.pth"
# - 'ConditionalGAN'            -> "Test_cGAN_Results/generator_final.pth"
# - 'cDCGAN'                    -> "Test_cDCGAN_Results/generator_final.pth"
# - 'cDCGAN_Physics'            -> "Test_cDCGAN_Physics_Results/generator_final.pth"
# - 'ACGAN'                     -> "Test_ACGAN_Results/generator_final.pth"
# - 'ACGAN_WGAN_GP'             -> "Test_ACGAN_WGAN_GP_Results/generator_final.pth"
# - 'ACGAN_WGAN_GP_DiffAug'     -> "Test_ACGAN_WGAN_GP_DiffAug_Results/generator_final.pth"
# - 'ACGAN_WGAN_GP_Physics'     -> "Test_ACGAN_WGAN_GP_Physics_Results/generator_final.pth"
#GENERATOR_CHECKPOINT = os.path.join(PROJECT_ROOT, "Test_Results", "Test_ACGAN_WGAN_GP_Physics_Results", "generator_final.pth")
GENERATOR_CHECKPOINT = os.path.join(PROJECT_ROOT, "Test_Results", "Test_ACGAN_WGAN_GP_Results", "generator_final.pth")

# 生成器类型 (必须匹配训练时的配置)
# 支持的类型: 'Baseline_GAN', 'ConditionalGAN', 'cDCGAN', 'ACGAN',
#            'ACGAN_WGAN_GP', 'ACGAN_WGAN_GP_DiffAug', 'ACGAN_WGAN_GP_Physics'
#GENERATOR_TYPE = 'ACGAN_WGAN_GP_Physics'
GENERATOR_TYPE = 'ACGAN_WGAN_GP'

# 实验参数
N_SHOTS = 20        # 每类使用的真实样本数量 (小样本设置)
M_GAN = 50          # 每类额外生成的 GAN 样本数量
TEST_RATIO = 0.2    # 测试集占总数据的比例
BATCH_SIZE = 64
EPOCHS = 50         # 分类器训练轮数
LEARNING_RATE = 0.001
SEED = 42           # 随机种子，确保可复现性
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 结果保存目录
RESULT_DIR = os.path.join(PROJECT_ROOT, "runs", "Comparative_Experiment")

# ============================================================
# 工具函数 (Helper Functions)
# ============================================================

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Info] Random seed set to {seed}")

class PadToSquare:
    """将图像填充为正方形"""
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return TF.pad(img, padding, 0, 'constant')

def get_generator_class(gen_type):
    """
    动态加载 GAN 生成器类,支持所有 GAN 模型类型。
    返回: (GeneratorClass, latent_dim, img_size, is_conditional)
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 定义模型配置: (目录, 文件名, 是否条件GAN)
    model_configs = {
        'Baseline_GAN': ('BaseLine_GAN', 'Baseline_GAN.py', False),
        'ConditionalGAN': ('ConditionalGAN', 'ConditionalGAN.py', True),
        'cDCGAN': ('cDCGAN', 'cDCGAN.py', True),
        'cDCGAN_Physics': ('cDCGAN', 'cDCGAN_Physics.py', True),
        'ACGAN': ('cDCGAN', 'ACGAN.py', True),
        'ACGAN_WGAN_GP': ('cDCGAN', 'ACGAN_WGAN_GP.py', True),
        'ACGAN_WGAN_GP_DiffAug': ('cDCGAN', 'ACGAN_WGAN_GP_DiffAug.py', True),
        'ACGAN_WGAN_GP_Physics': ('cDCGAN', 'ACGAN_WGAN_GP_Physics.py', True),
    }
    
    if gen_type not in model_configs:
        raise ValueError(f"Unknown generator type: {gen_type}. Supported types: {list(model_configs.keys())}")
    
    dir_name, filename, is_conditional = model_configs[gen_type]
    model_dir = os.path.join(project_root, dir_name)
    file_path = os.path.join(model_dir, filename)
    
    if not os.path.exists(file_path):
         raise FileNotFoundError(f"Generator file not found at: {file_path}")

    # 临时将模型目录加入 sys.path
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    
    spec = importlib.util.spec_from_file_location(gen_type, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[gen_type] = module
    spec.loader.exec_module(module)
    
    return module.Generator, module.LATENT_DIM, module.IMG_SIZE, is_conditional


def load_and_split_data(img_size=64):
    """
    加载数据并划分为:
    1. Test Set (固定)
    2. Train Pool (其余数据)
    3. Small Train Set (从 Train Pool 中每类抽取 N_SHOTS)
    
    Args:
        img_size: 图像大小,默认 64x64
    """
    if not os.path.exists(REAL_DATA_DIR):
        print(f"Error: Dataset not found at {REAL_DATA_DIR}")
        sys.exit(1)
        
    print(f"[Data] Loading Real Data from: {REAL_DATA_DIR}")
    print(f"[Data] Image Size: {img_size}x{img_size}")
    
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        PadToSquare(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    full_dataset = datasets.ImageFolder(REAL_DATA_DIR, transform=data_transforms)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"[Data] Classes: {class_names}")
    
    # 1. 划分 Test Set 和 Train Pool
    total_size = len(full_dataset)
    test_size = int(total_size * TEST_RATIO)
    train_pool_size = total_size - test_size
    
    train_pool_ds, test_ds = random_split(
        full_dataset, 
        [train_pool_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    print(f"[Data] Split: Train Pool={len(train_pool_ds)}, Test Set={len(test_ds)}")
    
    # 2. 从 Train Pool 中构建 N-Shot Small Train Set
    # 需要按类别抽取，random_split 无法保证每类均衡，所以我们需要手动处理
    # 这是一个比较棘手的问题，因为 random_split 后的 dataset 是 Subset，失去了 targets 属性
    # 我们通过遍历 indices 来获取 label
    
    # 获取 Train Pool 中所有样本的索引和标签
    train_pool_indices = train_pool_ds.indices
    train_pool_targets = [full_dataset.targets[i] for i in train_pool_indices]
    
    train_pool_targets = np.array(train_pool_targets)
    train_pool_indices = np.array(train_pool_indices)
    
    small_train_indices = []
    
    print(f"[Data] Sampling {N_SHOTS} shots per class from Train Pool...")
    for class_idx in range(num_classes):
        # 找到该类在 Train Pool 中的所有索引
        cls_mask = (train_pool_targets == class_idx)
        cls_indices = train_pool_indices[cls_mask]
        
        if len(cls_indices) < N_SHOTS:
            print(f"Warning: Class {class_names[class_idx]} has only {len(cls_indices)} samples in training pool, using all.")
            selected_indices = cls_indices
        else:
            # 随机抽取 N_SHOTS
            selected_indices = np.random.choice(cls_indices, N_SHOTS, replace=False)
        
        small_train_indices.extend(selected_indices)
        
    small_train_ds = Subset(full_dataset, small_train_indices)
    print(f"[Data] Small Train Set created: {len(small_train_ds)} samples total ({len(small_train_ds)//num_classes} per class avg)")
    
    return small_train_ds, test_ds, num_classes, class_names

def generate_gan_samples(num_samples_per_class, num_classes, gen_checkpoint, gen_type):
    """加载生成器并生成样本"""
    print(f"\n[GAN] Generating {num_samples_per_class} samples per class...")
    
    GeneratorClass, latent_dim, img_size, is_conditional = get_generator_class(gen_type)
    
    # 根据是否为条件 GAN 初始化生成器
    if is_conditional:
        netG = GeneratorClass(num_classes).to(DEVICE)
    else:
        netG = GeneratorClass().to(DEVICE)
    
    if not os.path.exists(gen_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {gen_checkpoint}")
        
    checkpoint = torch.load(gen_checkpoint, map_location=DEVICE)
    # 处理可能的 state_dict 键名不匹配问题 (e.g. 'module.' 前缀)
    if list(checkpoint.keys())[0].startswith('module.'):
        checkpoint = {k[7:]: v for k, v in checkpoint.items()}
        
    netG.load_state_dict(checkpoint)
    netG.eval()
    
    fake_images_list = []
    fake_labels_list = []
    
    chunk_size = 50
    
    with torch.no_grad():
        for c in range(num_classes):
            for _ in range(0, num_samples_per_class, chunk_size):
                current_batch = min(chunk_size, num_samples_per_class - _)
                z = torch.randn(current_batch, latent_dim).to(DEVICE)
                
                if is_conditional:
                    # 条件 GAN: 传递标签
                    labels = torch.full((current_batch,), c, dtype=torch.long).to(DEVICE)
                    imgs = netG(z, labels)
                else:
                    # 非条件 GAN: 不传递标签,生成的图像随机分配到各类
                    imgs = netG(z)
                    labels = torch.full((current_batch,), c, dtype=torch.long).to(DEVICE)
                
                fake_images_list.append(imgs.cpu())
                fake_labels_list.append(labels.cpu())
    
    all_fake_images = torch.cat(fake_images_list, dim=0)
    all_fake_labels = torch.cat(fake_labels_list, dim=0)
    
    print(f"[GAN] Generated {len(all_fake_images)} fake images.")
    
    # 封装为 Dataset
    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
        def __len__(self): return len(self.images)
        def __getitem__(self, idx): return self.images[idx], int(self.labels[idx])
        
    return FakeDataset(all_fake_images, all_fake_labels)

def train_classifier(train_dataset, test_dataset, num_classes, model_name):
    """训练分类器并返回最佳测试准确率"""
    print(f"\n[{model_name}] Starting Training...")
    print(f"[{model_name}] Training Set Size: {len(train_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 构建 ResNet18
    model = models.resnet18(pretrained=False)
    # 修改第一层以接受单通道输入
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # 修改全连接层适应类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_dataset):.4f} | Test Acc: {acc:.4f}")
            
    print(f"[{model_name}] Best Accuracy: {best_acc:.4f} (Time: {time.time()-start_time:.1f}s)")
    return best_acc

def main():
    set_seed(SEED)
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print(f"{'='*50}")
    print(f"SAR Comparative Experiment: Baseline vs GAN-Augmented")
    print(f"{'='*50}")
    print(f"[Config] GAN Type: {GENERATOR_TYPE}")
    
    # 0. 获取 GAN 模型的图像大小
    _, _, gan_img_size, _ = get_generator_class(GENERATOR_TYPE)
    print(f"[Config] GAN Image Size: {gan_img_size}x{gan_img_size}")
    
    # 1. 准备数据 (使用 GAN 的图像大小)
    small_real_train, test_ds, num_classes, class_names = load_and_split_data(img_size=gan_img_size)
    
    # 2. 训练 Baseline (仅真实小样本)
    baseline_acc = train_classifier(small_real_train, test_ds, num_classes, "Baseline (Real Only)")
    
    # 3. 生成 GAN 数据
    fake_dataset = generate_gan_samples(M_GAN, num_classes, GENERATOR_CHECKPOINT, GENERATOR_TYPE)
    
    # 4. 构建增强数据集 (Real + GAN)
    augmented_train = ConcatDataset([small_real_train, fake_dataset])
    print(f"\n[Data] Augmented Dataset: {len(small_real_train)} Real + {len(fake_dataset)} Fake = {len(augmented_train)} Total")
    
    # 5. 训练 Augmented Classifier
    aug_acc = train_classifier(augmented_train, test_ds, num_classes, "Augmented (Real + GAN)")
    
    # 6. 结果报告
    print(f"\n{'#'*50}")
    print(f"FINAL RESULTS")
    print(f"{'#'*50}")
    print(f"Tasks:")
    print(f"  - Real Shots per Class (N): {N_SHOTS}")
    print(f"  - GAN Shots per Class (M):  {M_GAN}")
    print(f"  - Test Set Size:            {len(test_ds)}")
    print(f"{'-'*50}")
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print(f"Augmented Accuracy: {aug_acc:.4f}")
    print(f"{'-'*50}")
    delta = aug_acc - baseline_acc
    print(f"Performance Gain: {delta:+.4f} ({delta*100:+.2f}%)")
    
    # 保存结果到文件
    report_path = os.path.join(RESULT_DIR, "experiment_report.txt")
    with open(report_path, "a") as f:
        f.write(f"\n\n=== Experiment at {time.ctime()} ===\n")
        f.write(f"Seed: {SEED} | N_Shots: {N_SHOTS} | M_GAN: {M_GAN}\n")
        f.write(f"Baseline: {baseline_acc:.4f}\n")
        f.write(f"Augmented: {aug_acc:.4f}\n")
        f.write(f"Gain: {delta:+.4f}\n")
        
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    main()
