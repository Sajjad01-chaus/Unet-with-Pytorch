import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import json
from PIL import Image

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, up_in_channels, skip_in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_in_channels, skip_in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(skip_in_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ColorEmbedding(nn.Module):
    def __init__(self, num_colors, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_colors, embedding_dim)

    def forward(self, color_idx):
        return self.embedding(color_idx)

class ConditionalUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=3, num_colors=10):
        super().__init__()
        self.color_embedding = ColorEmbedding(num_colors)

        self.inc = DoubleConv(n_channels + 1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(512, 512, 256)
        self.up2 = Up(256, 256, 128)
        self.up3 = Up(128, 128, 64)
        self.up4 = Up(64, 64, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x, color_idx):
        color_emb = self.color_embedding(color_idx)
        color_scalar = torch.mean(color_emb, dim=1, keepdim=True)
        B, _, H, W = x.shape
        color_map = color_scalar.view(B, 1, 1, 1).expand(-1, -1, H, W)
        x = torch.cat([x, color_map], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.outc(x))

class PolygonDataset(Dataset):
    def __init__(self, data, color_to_idx, data_path, mode='training', img_size=256):
        self.data = data
        self.color_to_idx = color_to_idx
        self.data_path = data_path
        self.mode = mode
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_img = Image.open(os.path.join(self.data_path, self.mode, 'inputs', sample['input_polygon'])).convert('L')
        target_img = Image.open(os.path.join(self.data_path, self.mode, 'outputs', sample['output_image'])).convert('RGB')
        input_tensor = self.transform(input_img)
        target_tensor = self.transform(target_img)
        color_idx = self.color_to_idx[sample['colour']]
        return input_tensor, torch.tensor(color_idx, dtype=torch.long), target_tensor

def load_dataset_info(data_path, mode='training'):
    with open(os.path.join(data_path, mode, 'data.json'), 'r') as f:
        data = json.load(f)
    unique_colors = sorted(set(sample['colour'] for sample in data))
    color_to_idx = {color: i for i, color in enumerate(unique_colors)}
    return data, color_to_idx

def create_data_loaders(data_path, batch_size=16, img_size=256, num_workers=2):
    train_data, color_to_idx = load_dataset_info(data_path, 'training')
    val_data, _ = load_dataset_info(data_path, 'validation')

    train_ds = PolygonDataset(train_data, color_to_idx, data_path, 'training', img_size)
    val_ds = PolygonDataset(val_data, color_to_idx, data_path, 'validation', img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, color_to_idx, len(color_to_idx)

def get_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)
    }
