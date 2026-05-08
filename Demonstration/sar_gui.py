"""
SAR Image Classifier & Generator GUI
=====================================
基于 ACGAN-WGAN-GP-Physics 模型的 SAR 图像分类与生成演示程序。

使用方法:
    python sar_gui.py

依赖:
    pip install PyQt5 torch torchvision Pillow
"""

import os
import sys
import time
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QStatusBar,
    QGroupBox, QComboBox, QGridLayout, QFrame,
)
from PyQt5.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt, QMimeData, pyqtSignal

# ============================================================
# Global Hyperparameters (与训练代码一致)
# ============================================================
LATENT_DIM = 100
IMG_SIZE = 64
IMG_CHANNELS = 1

# 路径 (相对于此脚本所在目录)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "Test_Results", "Test_ACGAN_WGAN_GP_Physics_Results")
GEN_WEIGHT_PATH = os.path.join(WEIGHT_DIR, "generator_final.pth")
DISC_WEIGHT_PATH = os.path.join(WEIGHT_DIR, "discriminator_final.pth")
CLASSIFIER_PATH = os.path.join(SCRIPT_DIR, "classifier.pth")
DATASET_ROOT = os.path.join(PROJECT_ROOT, "MSTAR", "PERSONAL_MSTAR", "15_DEG")
DEFAULT_SAVE_DIR = os.path.join(SCRIPT_DIR, "generated_samples")


# ============================================================
# Model Definitions (与 ACGAN_WGAN_GP_Physics.py 完全一致)
# ============================================================
class LogPhysicsLayer(nn.Module):
    def __init__(self, num_classes, init_sigma=0.05):
        super(LogPhysicsLayer, self).__init__()
        val = np.log(init_sigma)
        self.log_sigma = nn.Parameter(torch.full((num_classes, 1, 1, 1), val))

    def forward(self, log_reflectance, labels):
        batch_log_sigma = self.log_sigma[labels]
        sigma = torch.exp(batch_log_sigma)
        noise = torch.randn_like(log_reflectance) * sigma
        return log_reflectance + noise


class Generator(nn.Module):
    def __init__(self, class_count):
        super(Generator, self).__init__()
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
        )
        self.physics_layer = LogPhysicsLayer(num_classes=class_count)
        self.final_act = nn.Tanh()

    def forward(self, noise_vec, cond_labels):
        label_embed = self.label_embedding(cond_labels)
        combined = torch.cat([noise_vec, label_embed], dim=1)
        combined = combined.unsqueeze(2).unsqueeze(3)
        log_reflectance = self.net(combined)
        noisy_log = self.physics_layer(log_reflectance, cond_labels)
        return self.final_act(noisy_log)


class Discriminator(nn.Module):
    def __init__(self, class_count):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(class_count, IMG_SIZE * IMG_SIZE)
        self.features = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS + 1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.adv_layer = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, image_tensor, cond_labels):
        label_embed = self.label_embedding(cond_labels)
        label_img = label_embed.view(-1, 1, IMG_SIZE, IMG_SIZE)
        combined = torch.cat([image_tensor, label_img], dim=1)
        features = self.features(combined)
        validity = self.adv_layer(features).view(-1, 1)
        return validity


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# ============================================================
# Lightweight CNN Classifier (独立训练的分类器)
# ============================================================
class SARClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# 图像预处理 (与训练时 transform 完全一致)
# ============================================================
preprocess = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def tensor_to_qpixmap(tensor_1ch):
    """将 [-1,1] 范围的单通道 tensor 转为 QPixmap 用于显示。"""
    img = tensor_1ch.detach().cpu().squeeze().numpy()
    img = ((img + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    h, w = img.shape
    qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())


def tensor_to_pil(tensor_1ch):
    """将 [-1,1] 范围的单通道 tensor 转为 PIL Image。"""
    img = tensor_1ch.detach().cpu().squeeze().numpy()
    img = ((img + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="L")


# ============================================================
# GUI 主窗口
# ============================================================
class SARGuiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAR Image Classifier & Generator")
        self.setMinimumSize(900, 620)
        self.setAcceptDrops(False)

        # 模型相关
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen_model = None
        self.disc_model = None
        self.classifier = None
        self.num_classes = 0
        self.class_names = []

        # 当前状态
        self.input_tensor = None       # 预处理后的输入图像 tensor [1, 1, 64, 64]
        self.predicted_class = -1
        self.generated_images = []     # list of tensor [1, 1, 64, 64]

        self._init_ui()
        self._load_models()

    # ----------------------------------------------------------
    # UI 初始化
    # ----------------------------------------------------------
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ---- 上半部分: 图像显示区域 ----
        img_layout = QHBoxLayout()

        # 左侧: 原始图像
        left_group = QGroupBox("原始图像")
        left_layout = QVBoxLayout(left_group)
        self.original_label = DropImageLabel("拖拽或点击加载图像")
        self.original_label.setFixedSize(280, 280)
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet(
            "QLabel { background-color: #2b2b2b; color: #888; "
            "border: 2px dashed #555; font-size: 14px; }"
        )
        self.original_label.clicked.connect(self._load_image_dialog)
        left_layout.addWidget(self.original_label, alignment=Qt.AlignCenter)

        self.pred_class_label = QLabel("预测类别: --")
        self.pred_class_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        left_layout.addWidget(self.pred_class_label)

        self.confidence_label = QLabel("置信度: --")
        self.confidence_label.setStyleSheet("font-size: 13px;")
        left_layout.addWidget(self.confidence_label)

        img_layout.addWidget(left_group)

        # 右侧: 生成图像 (2x2)
        right_group = QGroupBox("生成图像 (2×2)")
        right_layout = QGridLayout(right_group)
        self.gen_labels = []
        for row in range(2):
            for col in range(2):
                lbl = QLabel()
                lbl.setFixedSize(128, 128)
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet(
                    "QLabel { background-color: #2b2b2b; "
                    "border: 1px solid #444; }"
                )
                right_layout.addWidget(lbl, row, col)
                self.gen_labels.append(lbl)
        img_layout.addWidget(right_group)

        main_layout.addLayout(img_layout)

        # ---- 类别选择 (用于手动指定生成类别) ----
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("选择类别:"))
        self.class_combo = QComboBox()
        self.class_combo.setMinimumWidth(150)
        class_layout.addWidget(self.class_combo)
        class_layout.addStretch()
        main_layout.addLayout(class_layout)

        # ---- 按钮区域 ----
        btn_layout = QHBoxLayout()

        self.btn_load = QPushButton("加载图像")
        self.btn_load.setMinimumHeight(38)
        self.btn_load.clicked.connect(self._load_image_dialog)
        btn_layout.addWidget(self.btn_load)

        self.btn_classify = QPushButton("分类预测")
        self.btn_classify.setMinimumHeight(38)
        self.btn_classify.clicked.connect(self._classify_image)
        self.btn_classify.setEnabled(False)
        btn_layout.addWidget(self.btn_classify)

        self.btn_generate = QPushButton("生成虚假样本")
        self.btn_generate.setMinimumHeight(38)
        self.btn_generate.clicked.connect(self._generate_samples)
        self.btn_generate.setEnabled(False)
        btn_layout.addWidget(self.btn_generate)

        self.btn_save = QPushButton("保存生成图像")
        self.btn_save.setMinimumHeight(38)
        self.btn_save.clicked.connect(self._save_images)
        self.btn_save.setEnabled(False)
        btn_layout.addWidget(self.btn_save)

        main_layout.addLayout(btn_layout)

        # ---- 状态栏 ----
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    # ----------------------------------------------------------
    # 模型加载
    # ----------------------------------------------------------
    def _load_models(self):
        self.status_bar.showMessage("正在加载模型...")

        # 从分类器权重文件中读取类别信息
        if os.path.isfile(CLASSIFIER_PATH):
            try:
                ckpt = torch.load(CLASSIFIER_PATH, map_location="cpu", weights_only=False)
                self.class_names = ckpt["class_names"]
                self.num_classes = ckpt["num_classes"]
            except Exception:
                self._infer_classes_from_dataset()
        else:
            self._infer_classes_from_dataset()

        # 填充类别下拉框
        self.class_combo.addItems(self.class_names)

        # 加载分类器
        if os.path.isfile(CLASSIFIER_PATH):
            try:
                ckpt = torch.load(CLASSIFIER_PATH, map_location=self.device, weights_only=False)
                self.classifier = SARClassifier(self.num_classes).to(self.device)
                self.classifier.load_state_dict(ckpt["model_state_dict"])
                self.classifier.eval()
            except Exception as e:
                QMessageBox.warning(self, "分类器加载失败", str(e))

        # 加载 GAN 模型
        if not os.path.isfile(GEN_WEIGHT_PATH):
            QMessageBox.critical(self, "错误",
                                 f"生成器权重文件不存在:\n{GEN_WEIGHT_PATH}")
            self.status_bar.showMessage("模型加载失败")
            return

        try:
            self.gen_model = Generator(self.num_classes).to(self.device)
            self.gen_model.load_state_dict(
                torch.load(GEN_WEIGHT_PATH, map_location=self.device, weights_only=True))
            self.gen_model.eval()

            # 判别器为可选加载
            if os.path.isfile(DISC_WEIGHT_PATH):
                self.disc_model = Discriminator(self.num_classes).to(self.device)
                self.disc_model.load_state_dict(
                    torch.load(DISC_WEIGHT_PATH, map_location=self.device, weights_only=True))
                self.disc_model.eval()

            self.btn_generate.setEnabled(True)

            dev_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
            self.status_bar.showMessage(
                f"模型加载完成 | 设备: {dev_name} | 类别数: {self.num_classes}")
        except Exception as e:
            QMessageBox.critical(self, "模型加载错误", str(e))
            self.status_bar.showMessage("模型加载失败")

    def _infer_classes_from_dataset(self):
        if os.path.isdir(DATASET_ROOT):
            self.class_names = sorted([
                d for d in os.listdir(DATASET_ROOT)
                if os.path.isdir(os.path.join(DATASET_ROOT, d))
            ])
            self.num_classes = len(self.class_names)
        else:
            self.class_names = ["2S1", "BRDM_2", "D7", "T62", "ZIL131", "ZSU_23_4"]
            self.num_classes = 6

    # ----------------------------------------------------------
    # 图像加载
    # ----------------------------------------------------------
    def _load_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 SAR 图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;所有文件 (*)")
        if path:
            self._load_image_from_path(path)

    def _load_image_from_path(self, path):
        try:
            img = Image.open(path)
            tensor = preprocess(img).unsqueeze(0).to(self.device)  # [1, 1, 64, 64]
            self.input_tensor = tensor

            # 显示图像
            pixmap = tensor_to_qpixmap(tensor.squeeze(0))
            scaled = pixmap.scaled(260, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.original_label.setPixmap(scaled)
            self.original_label.setStyleSheet(
                "QLabel { background-color: #1a1a1a; border: 2px solid #4a9eff; }")

            self.btn_classify.setEnabled(True)
            self.btn_generate.setEnabled(True)

            # 自动分类
            self._classify_image()

            self.status_bar.showMessage(f"图像已加载: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "加载失败", f"无法加载图像:\n{e}")

    # ----------------------------------------------------------
    # 分类预测
    # ----------------------------------------------------------
    def _classify_image(self):
        if self.input_tensor is None or self.classifier is None:
            return

        self.status_bar.showMessage("正在分类...")
        QApplication.processEvents()

        try:
            with torch.no_grad():
                logits = self.classifier(self.input_tensor)  # [1, num_classes]
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            best_cls = int(np.argmax(probs))
            self.predicted_class = best_cls

            cls_name = self.class_names[best_cls] if best_cls < len(self.class_names) else str(best_cls)
            conf_pct = probs[best_cls] * 100

            self.pred_class_label.setText(f"预测类别: {cls_name} (索引 {best_cls})")
            self.confidence_label.setText(f"置信度: {conf_pct:.1f}%")

            # 更新下拉框
            self.class_combo.setCurrentIndex(best_cls)

            self.status_bar.showMessage(
                f"预测类别: {cls_name} | 置信度: {conf_pct:.1f}%")

        except Exception as e:
            QMessageBox.warning(self, "分类错误", str(e))
            self.status_bar.showMessage("分类失败")

    # ----------------------------------------------------------
    # 虚假样本生成
    # ----------------------------------------------------------
    def _generate_samples(self):
        if self.gen_model is None:
            return

        self.status_bar.showMessage("正在生成虚假样本...")
        QApplication.processEvents()

        try:
            # 使用下拉框选择的类别
            cls_idx = self.class_combo.currentIndex()
            num_gen = 4

            with torch.no_grad():
                noise = torch.randn(num_gen, LATENT_DIM, device=self.device)
                labels = torch.full((num_gen,), cls_idx, dtype=torch.long, device=self.device)
                fake_imgs = self.gen_model(noise, labels)  # [4, 1, 64, 64]

            self.generated_images = [fake_imgs[i:i+1] for i in range(num_gen)]

            # 显示 2x2 网格
            for i, lbl in enumerate(self.gen_labels):
                pixmap = tensor_to_qpixmap(fake_imgs[i])
                scaled = pixmap.scaled(126, 126, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl.setPixmap(scaled)

            cls_name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else str(cls_idx)
            self.btn_save.setEnabled(True)
            self.status_bar.showMessage(f"已生成 {num_gen} 张 {cls_name} 类别虚假样本")

        except Exception as e:
            QMessageBox.warning(self, "生成错误", str(e))
            self.status_bar.showMessage("生成失败")

    # ----------------------------------------------------------
    # 保存生成图像
    # ----------------------------------------------------------
    def _save_images(self):
        if not self.generated_images:
            QMessageBox.information(self, "提示", "没有可保存的生成图像")
            return

        save_dir = QFileDialog.getExistingDirectory(
            self, "选择保存目录", DEFAULT_SAVE_DIR)
        if not save_dir:
            return

        try:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cls_idx = self.class_combo.currentIndex()
            cls_name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else str(cls_idx)

            saved_paths = []
            for i, img_tensor in enumerate(self.generated_images):
                fname = f"fake_class{cls_idx}_{cls_name}_{timestamp}_{i}.png"
                fpath = os.path.join(save_dir, fname)
                pil_img = tensor_to_pil(img_tensor)
                pil_img.save(fpath)
                saved_paths.append(fpath)

            self.status_bar.showMessage(
                f"已保存 {len(saved_paths)} 张图像到: {save_dir}")
            QMessageBox.information(
                self, "保存成功",
                f"已保存 {len(saved_paths)} 张图像到:\n{save_dir}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))


# ============================================================
# 支持拖拽/点击的 QLabel
# ============================================================
class DropImageLabel(QLabel):
    """支持拖拽图像文件和点击触发加载的自定义 QLabel。"""
    clicked = pyqtSignal()

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                "QLabel { background-color: #1a3a5c; color: #4a9eff; "
                "border: 2px dashed #4a9eff; font-size: 14px; }")

    def dragLeaveEvent(self, event):
        if self.pixmap() is None:
            self.setStyleSheet(
                "QLabel { background-color: #2b2b2b; color: #888; "
                "border: 2px dashed #555; font-size: 14px; }")

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(
            "QLabel { background-color: #2b2b2b; color: #888; "
            "border: 2px dashed #555; font-size: 14px; }")
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
                # 找到主窗口实例并调用加载方法
                main_win = self.window()
                if isinstance(main_win, SARGuiApp):
                    main_win._load_image_from_path(path)


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置全局样式
    app.setStyleSheet("""
        QMainWindow { background-color: #f0f0f0; }
        QGroupBox {
            font-weight: bold;
            font-size: 13px;
            border: 1px solid #ccc;
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 14px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 4px;
        }
        QPushButton {
            background-color: #4a9eff;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 13px;
            padding: 6px 16px;
        }
        QPushButton:hover { background-color: #3a8eef; }
        QPushButton:pressed { background-color: #2a7edf; }
        QPushButton:disabled { background-color: #aaa; }
        QComboBox { padding: 4px 8px; border-radius: 4px; border: 1px solid #ccc; }
        QStatusBar { font-size: 12px; }
    """)

    window = SARGuiApp()
    window.show()
    sys.exit(app.exec_())
