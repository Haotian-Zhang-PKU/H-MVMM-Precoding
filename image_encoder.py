# image_encoder.py
# ─────────────────────────────────────────────────────────────
# 提取 RGB 图像特征，用于与 LiDAR Backbone2D 的特征进行融合
# 依赖: torch 2.x, torchvision 0.17+, Pillow, numpy
# ─────────────────────────────────────────────────────────────
import os
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tvm

# ─────────────────────────────────────────────────────────────
# 设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Image encoder running on: {device}')

# ─────────────────────────────────────────────────────────────
def load_resnet18(pretrained: bool = True):
    """
    Returns a torchvision.models.resnet18 configured correctly for both
    new (>=0.13) and old (<0.13) torchvision versions.
    """
    try:  # new API
        weights = (
            tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        return tvm.resnet18(weights=weights)
    except AttributeError:  # old API
        return tvm.resnet18(pretrained=pretrained)
    
# 数据集
class ImageFolderRGB(Dataset):
    """
    将目录下的所有 *.jpg|*.png|*.jpeg 作为样本。
    transform: torchvision.transforms
    """
    IMG_EXT = ('*.jpg', '*.png', '*.jpeg', '*.bmp')

    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.files = []
        for ext in self.IMG_EXT:
            self.files += glob.glob(os.path.join(root_dir, ext))
        self.files.sort()
        self.transform = transform or T.Compose([
            T.Resize(256),
            T.CenterCrop(960),
            T.ToTensor(),                      # 转 [0,1]
            T.Normalize([0.485, 0.456, 0.406], # ImageNet 统计量
                        [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img)
        return {
            'image': tensor,             # (3, H, W)
            'file_name': os.path.basename(img_path)
        }


def create_image_loader(root_dir, batch_size=4, num_workers=4, shuffle=False):
    ds = ImageFolderRGB(root_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)

# ─────────────────────────────────────────────────────────────
# 主干网络 + FPN  (与 LiDAR Backbone2D 结构保持一致的输出分辨率)
class ResNet18Backbone(nn.Module):
    """
    以 torchvision ResNet-18 为基础，输出三个尺度的特征 (C2, C3, C4)。
    """
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = load_resnet18(pretrained)

        # stem
        self.stem = nn.Sequential(
            resnet.conv1,   # /2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # /4
        )
        # ResNet 的四个 stage
        self.layer1 = resnet.layer1          # /4
        self.layer2 = resnet.layer2          # /8
        self.layer3 = resnet.layer3          # /16
        # layer4 (/32) 通常太小，在此省略，保持与 LiDAR 金字塔深度一致

    def forward(self, x):
        x = self.stem(x)     # /4
        c2 = self.layer1(x)  # /4
        c3 = self.layer2(c2) # /8
        c4 = self.layer3(c3) # /16
        return [c2, c3, c4]  # 与 LiDAR: [C2(1/2), C3(1/4), C4(1/8)] 尺度相对应


class FPN(nn.Module):
    """
    与 LiDAR 版完全相同，见 lidarencoder.py
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        # 1×1 Conv 先将每层通道压到统一的 out_channels
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        # 3×3 Conv “平滑”卷积，去除上采样带来的棋盘格噪声
        self.smooth  = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(len(in_channels_list) - 1)  # 最深层 P4 不需要再平滑
        ])
        # 默认使用 nearest-neighbor 上采样，可替换成 F.interpolate
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, feats):              # feats = [C2, C3, C4] 由浅到深
        lat = [l(f) for l, f in zip(self.lateral, feats)]   # 每层 1×1
        # 自顶向下：先放入最深层 (C4→P4) 作为初始
        p = [lat[-1]]                                       # len=1
        # 反向遍历：C3 → C2
        for i in range(len(lat) - 2, -1, -1):
            up = self.upsample(p[0])                        # 上采样到上一层尺寸
            # 由于整除不一定精确，保险起见再用 interpolate 调整到完全对齐
            if up.shape[2:] != lat[i].shape[2:]:
                up = F.interpolate(up, size=lat[i].shape[2:], mode='nearest')
            # 横向相加（element-wise fusion）
            m  = lat[i] + up                                # 融合语义
            m  = self.smooth[i](m)                          # 3×3 平滑卷积
            p.insert(0, m)                                  # 头插保持顺序
        return p    # [P2, P3, P4] 与输入一一对应

# ─────────────────────────────────────────────────────────────
class ImageFeatureNet(nn.Module):
    """
    综合 ResNet18 + FPN，再用 1×1 conv 压到可配置的 channel (默认 128)。
    输出与 LiDAR Backbone2D 的最高分辨率 (1/2) 对齐，可直接 cat。
    """
    def __init__(self, out_channels=128, pretrained=True):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained)
        self.fpn      = FPN([64, 128, 256], out_channels=256)
        self.reduce   = nn.Conv2d(256, out_channels, 1)

    def forward(self, x):
        p2, p3, p4 = self.fpn(self.backbone(x))   # p2: /4, p3: /8, p4: /16
        feat_img   = self.reduce(p2)              # 与 LiDAR features[0] 尺度一致
        return feat_img        # shape: (B, out_channels, H/4, W/4)

# ─────────────────────────────────────────────────────────────
# 可选: 参数/算力相近但精度更高的主干
# 1. MobileNetV3-Large (≈5.4 M params, 219 MFLOPs @224)  > ResNet-18 精度
# 2. EfficientNet-B0   (≈5.3 M params, 390 MFLOPs)        也可替换
# 替换方法:
#   backbone = tvm.mobilenet_v3_large(weights='IMAGENET1K_V2')  # etc.
# 然后选取其中层特征即可，接口保持一致即可拼接。

# ─────────────────────────────────────────────────────────────
# demo
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='/data2/Haotiandata/MM_Foundation_Model/test_image_encoder/test_image')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--out_channels', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='facades')
    args = parser.parse_args()
    # print('dataset =', args.dataset)

    loader = create_image_loader(args.img_dir, batch_size=args.batch_size)
    net = ImageFeatureNet(out_channels=args.out_channels).to(device).eval()

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)          # (B,3,H,W)
            feats = net(imgs)                         # (B,128,H/4,W/4)
            print(f"Image feats shape: {feats.shape}")
            break  # demo 只跑一批