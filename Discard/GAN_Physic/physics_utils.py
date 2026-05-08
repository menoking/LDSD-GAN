"""
SAR物理感知GAN - 物理约束工具模块

本模块实现了SAR图像的物理约束损失函数，用于提升GAN生成图像的物理真实性。
基于Φ-GAN的研究思路，实现以下物理约束：
1. 频域损失 - 确保频谱特性一致
2. 散斑噪声统计损失 - 保证散斑噪声符合SAR成像特性
3. 对比度损失 - 维持目标与背景的合理对比度
4. 梯度统计损失 - 保持SAR特有的边缘和纹理特征
"""

import torch
import torch.nn.functional as F
import numpy as np


# ============================================================
# 辅助工具：转换至强度空间 [0, 1]
# ============================================================
def to_intensity_space(imgs: torch.Tensor) -> torch.Tensor:
    """将 [-1, 1] 范围的 GAN 输出映射至 [0, 1] 强度空间进行物理计算"""
    return (imgs + 1.0) / 2.0


# ============================================================
# 频域损失 (Frequency Domain Loss) - 增强稳定性
# ============================================================
def frequency_domain_loss(fake_imgs: torch.Tensor, real_imgs: torch.Tensor) -> torch.Tensor:
    """计算生成图像与真实图像在频域的差异"""
    # 转换到强度空间更有利于频域分析
    f_imgs = to_intensity_space(fake_imgs)
    r_imgs = to_intensity_space(real_imgs)
    
    fake_fft = torch.fft.fft2(f_imgs, dim=(-2, -1))
    real_fft = torch.fft.fft2(r_imgs, dim=(-2, -1))
    
    # 使用幅度谱，并添加较大部分的 eps
    fake_magnitude = torch.abs(fake_fft) + 1e-6
    real_magnitude = torch.abs(real_fft) + 1e-6
    
    # 对数尺度压缩动态范围
    fake_log = torch.log(fake_magnitude)
    real_log = torch.log(real_magnitude)
    
    # 使用 Smooth L1 减少梯度爆炸风险
    return F.smooth_l1_loss(fake_log, real_log)


# ============================================================
# 散斑噪声统计损失 (Speckle Statistics Loss) - 修复爆炸问题
# ============================================================
def speckle_statistics_loss(fake_imgs: torch.Tensor, real_imgs: torch.Tensor, 
                           kernel_size: int = 7) -> torch.Tensor:
    """
    计算散斑噪声的统计特性差异。
    修复：使用强度空间计算，并采用 log 域对比，避免直接除法导致的数值爆炸。
    """
    f_imgs = to_intensity_space(fake_imgs)
    r_imgs = to_intensity_space(real_imgs)
    
    def get_stats(imgs):
        mean = F.avg_pool2d(imgs, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        mean_sq = F.avg_pool2d(imgs ** 2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        var = F.relu(mean_sq - mean ** 2) # 确保非负
        std = torch.sqrt(var + 1e-6)
        return mean, std

    f_mean, f_std = get_stats(f_imgs)
    r_mean, r_std = get_stats(r_imgs)
    
    # SAR 的 CV 是 std / mean。
    # 为了数值稳定，我们对比 log(std + eps) - log(mean + eps)
    f_log_cv = torch.log(f_std + 1e-6) - torch.log(f_mean + 1e-6)
    r_log_cv = torch.log(r_std + 1e-6) - torch.log(r_mean + 1e-6)
    
    # 使用 Smooth L1 限制极端梯度
    return F.smooth_l1_loss(f_log_cv, r_log_cv)


# ============================================================
# 目标-背景对比度损失 (Target-Background Contrast Loss) - 增加目标值
# ============================================================
def contrast_loss(fake_imgs: torch.Tensor, real_imgs: torch.Tensor, window_size: int = 8) -> torch.Tensor:
    """
    不再盲目最大化对比度，而是让生成图像的局部对比度水平接近真实图像。
    """
    f_imgs = to_intensity_space(fake_imgs)
    r_imgs = to_intensity_space(real_imgs)
    
    def get_local_std(imgs):
        mean = F.avg_pool2d(imgs, kernel_size=window_size, stride=window_size // 2, padding=window_size // 2)
        mean_sq = F.avg_pool2d(imgs ** 2, kernel_size=window_size, stride=window_size // 2, padding=window_size // 2)
        return torch.sqrt(F.relu(mean_sq - mean ** 2) + 1e-6)

    f_contrast = get_local_std(f_imgs)
    r_contrast = get_local_std(r_imgs)
    
    return F.smooth_l1_loss(f_contrast, r_contrast)


# ============================================================
# 梯度统计损失 (Gradient Statistics Loss) - 增强稳定性
# ============================================================
def gradient_statistics_loss(fake_imgs: torch.Tensor, real_imgs: torch.Tensor) -> torch.Tensor:
    """保持梯度分布一致"""
    f_imgs = to_intensity_space(fake_imgs)
    r_imgs = to_intensity_space(real_imgs)
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=f_imgs.dtype, device=f_imgs.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=f_imgs.dtype, device=f_imgs.device).view(1, 1, 3, 3)
    
    f_grad = torch.sqrt(F.conv2d(f_imgs, sobel_x, padding=1)**2 + F.conv2d(f_imgs, sobel_y, padding=1)**2 + 1e-6)
    r_grad = torch.sqrt(F.conv2d(r_imgs, sobel_x, padding=1)**2 + F.conv2d(r_imgs, sobel_y, padding=1)**2 + 1e-6)
    
    return F.smooth_l1_loss(f_grad, r_grad)


# ============================================================
# 综合物理损失 (Combined Physics Loss)
# ============================================================
def combined_physics_loss(fake_imgs: torch.Tensor, real_imgs: torch.Tensor,
                         lambda_freq: float = 0.1,
                         lambda_speckle: float = 0.05,
                         lambda_contrast: float = 0.1,
                         lambda_gradient: float = 0.05) -> dict:
    
    # 所有的损失现在都基于对比或 Smooth L1，大大增强了稳定性
    freq_l = frequency_domain_loss(fake_imgs, real_imgs)
    speckle_l = speckle_statistics_loss(fake_imgs, real_imgs)
    contrast_l = contrast_loss(fake_imgs, real_imgs)
    gradient_l = gradient_statistics_loss(fake_imgs, real_imgs)
    
    total_physics_loss = (lambda_freq * freq_l + 
                         lambda_speckle * speckle_l + 
                         lambda_contrast * contrast_l + 
                         lambda_gradient * gradient_l)
    
    return {
        'total': total_physics_loss,
        'freq': freq_l,
        'speckle': speckle_l,
        'contrast': contrast_l,
        'gradient': gradient_l
    }


# ============================================================
# 可视化工具
# ============================================================
def compute_physics_metrics(imgs: torch.Tensor) -> dict:
    with torch.no_grad():
        f_imgs = to_intensity_space(imgs)
        
        # 统计
        mean = torch.mean(f_imgs).item()
        std = torch.std(f_imgs).item()
        
        # 梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=f_imgs.dtype, device=f_imgs.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=f_imgs.dtype, device=f_imgs.device).view(1, 1, 3, 3)
        grad = torch.sqrt(F.conv2d(f_imgs, sobel_x, padding=1)**2 + F.conv2d(f_imgs, sobel_y, padding=1)**2 + 1e-6)
        
    return {
        'mean': mean,
        'std': std,
        'grad_mean': torch.mean(grad).item()
    }

