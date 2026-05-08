import os
import numpy as np
import argparse

from torchvision.utils import save_image

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import device
import torch

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./images", exist_ok=True)  # 创建images文件夹

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="numbers of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)


class Generator(nn.Module):  # 继承自nn.Module
    def __init__(self):
        super(Generator, self).__init__()  # 调用父类初始化

        def block(in_feat, out_feat, normalize=True):  # 每层分块：输入维度，输出维度，是否归一化
            layers = [nn.Linear(in_feat, out_feat)]  # 全连接层
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))  # 批归一化，动量参数为0.8
            return layers  # 返回整个层的列表

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),  # 解包列表为对象
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),  # product（乘积），将输入序列中的所有元素连乘起来。
            nn.Tanh()  # 非线性激活
        )

    def forward(self, z):
        img = self.model(z)  # 调用自身的方法创建img变量
        img = img.view(img.size(0), *img_shape)  # 维度重构,img.size(0)取的是当前批次数
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # -1作占位符，自动计算维度
        validity = self.model(img_flat)  # 展平后的图像传入模型得到有效分数
        return validity


# 定义损失函数：二元交叉熵损失函数
adversarial_loss = nn.BCELoss()

# 实例化
generator = Generator()
discriminator = Discriminator()

# 选择计算设备
generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)

# 数据集处理:MNIST数据集，输入批次，shuffle打乱数据
os.makedirs("../DataSet_MNIST", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../DataSet_MNIST',
                   train=True,
                   download=True,
                   transform=transforms.Compose(
                       [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
                   ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# 优化器设计
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 定义浮点运算设备
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

#----------------------------------
#训练过程
#----------------------------------
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # 真假标签
        Real = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        Fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # 真实样本
        real_img = Variable(imgs.type(Tensor))

        # 生成器梯度清零
        optimizer_G.zero_grad()
        # 生成噪声：要求符合正态分布（均值0方差1），形状为batch_size及latent_dim
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        # 生成假样本
        fake_img = generator(z)

        # 生成器损失函数计算
        g_loss = adversarial_loss(discriminator(fake_img),Real)
        # 生成器反向传播
        g_loss.backward()
        # 生成器更新参数
        optimizer_G.step()

        # 判别器梯度清零
        optimizer_D.zero_grad()
        # 判别真样本的能理
        Real_loss = adversarial_loss(discriminator(real_img), Real)
        # 判别假样本的能力
        Fake_loss = adversarial_loss(discriminator(fake_img.detach()), Fake)
        # 判别器损失函数
        d_loss = (Real_loss + Fake_loss) / 2
        # 反向传播
        d_loss.backward()
        # 判别器更新参数
        optimizer_D.step()

        print("[Epoch: %d/%d ] [Batch: %d/%d ] [D loss: %f ] [G loss: %f ]"
              % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(fake_img.data[:25], "./images/%d.png" % batches_done, nrow=5, normalize=True)