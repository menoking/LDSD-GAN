"""
测试脚本:验证所有 GAN 模型是否能被正确加载
"""
import os
import sys

# 添加 Evaluation_Tools 到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from comparative_experiment import get_generator_class

# 测试所有支持的 GAN 类型
gan_types = [
    'Baseline_GAN',
    'ConditionalGAN',
    'cDCGAN',
    'ACGAN',
    'ACGAN_WGAN_GP',
    'ACGAN_WGAN_GP_DiffAug',
    'ACGAN_WGAN_GP_Physics'
]

print("=" * 60)
print("测试 GAN 模型加载功能")
print("=" * 60)

for gan_type in gan_types:
    try:
        GeneratorClass, latent_dim, img_size, is_conditional = get_generator_class(gan_type)
        cond_str = "Conditional" if is_conditional else "Unconditional"
        print(f"[OK] {gan_type:30s} - {cond_str:15s} | LATENT_DIM={latent_dim:3d} | IMG_SIZE={img_size:3d}")
    except Exception as e:
        print(f"[FAIL] {gan_type:30s} - Error: {str(e)}")

print("=" * 60)
print("Test Complete!")
print("=" * 60)
