"""Visual agent variant: XceptionNet over the RGB face crop plus five forensic channels the
network computes on the device from the same crop: an error-level-analysis map (luma
re-quantised through the 8x8 JPEG DCT at the quality-90 luminance table), the log-magnitude
Fourier spectrum, an 8-neighbour local-binary-pattern code, and the Cb and Cr chroma planes.
The biometric-quality agent already consumes a Laplacian sharpness map and a high-frequency
residual, so those cues are left to it. The input contract is the released visual agent's
(normalised 3-channel crop, frame mean with horizontal-flip test-time augmentation)."""
import math

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 299
EXTRA = 5
# stored crops are in OpenCV channel order (B, G, R)
_LUMA = (0.114, 0.587, 0.299)
_JPEG_LUMA = torch.tensor([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
                           [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                           [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=torch.float32)


def _dct_matrix():
    n = torch.arange(8, dtype=torch.float32)
    D = torch.cos(math.pi * (2 * n[None, :] + 1) * n[:, None] / 16) * math.sqrt(2 / 8)
    D[0] /= math.sqrt(2)
    return D


class ForensicChannels(nn.Module):
    def __init__(self, quality=90):
        super().__init__()
        scale = (200 - 2 * quality) / 100.0 if quality >= 50 else 5000.0 / quality / 100.0
        self.register_buffer("Q", torch.clamp(torch.round(_JPEG_LUMA * scale), 1, 255))
        self.register_buffer("D", _dct_matrix())
        self.register_buffer("luma_w", torch.tensor(_LUMA).view(1, 3, 1, 1))

    def ela(self, y):
        """|luma - JPEG-requantised luma| on the 8x8 DCT grid, scaled like the CPU definition."""
        B, _, H, W = y.shape; ph, pw = (-H) % 8, (-W) % 8
        x = F.pad(y * 255.0 - 128.0, (0, pw, 0, ph), mode="replicate")
        blocks = F.unfold(x, kernel_size=8, stride=8).transpose(1, 2).reshape(-1, 8, 8)
        coef = self.D @ blocks @ self.D.T
        coef = torch.round(coef / self.Q) * self.Q
        rec = (self.D.T @ coef @ self.D).reshape(B, -1, 64).transpose(1, 2)
        rec = F.fold(rec, output_size=(H + ph, W + pw), kernel_size=8, stride=8)[:, :, :H, :W] + 128.0
        return torch.clamp((y * 255.0 - rec).abs() / 255.0 * 16.0, 0.0, 1.0) * 2.0 - 1.0

    @staticmethod
    def spectrum(y):
        mag = torch.log1p(torch.fft.fftshift(torch.fft.fft2(y - y.mean(dim=(2, 3), keepdim=True)), dim=(-2, -1)).abs())
        lo = mag.amin(dim=(2, 3), keepdim=True); hi = mag.amax(dim=(2, 3), keepdim=True)
        return (mag - lo) / (hi - lo + 1e-6) * 2.0 - 1.0

    @staticmethod
    def lbp(y):
        p = F.pad(y, (1, 1, 1, 1), mode="replicate"); H, W = y.shape[-2:]; code = torch.zeros_like(y)
        for bit, (dy, dx) in enumerate(((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))):
            code = code + (p[:, :, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W] >= y).to(y.dtype) * (2 ** bit)
        return code / 127.5 - 1.0

    def forward(self, rgb01):
        """rgb01: B x 3 x H x W in [0, 1], stored (B, G, R) order -> B x 5 x H x W in [-1, 1]."""
        y = (rgb01 * self.luma_w).sum(dim=1, keepdim=True)
        b, r = rgb01[:, 0:1], rgb01[:, 2:3]
        cr = torch.clamp(0.713 * (r - y) + 0.5, 0, 1) * 2.0 - 1.0; cb = torch.clamp(0.564 * (b - y) + 0.5, 0, 1) * 2.0 - 1.0
        return torch.cat([self.ela(y), self.spectrum(y), self.lbp(y), cb, cr], dim=1)


def to_input(img):
    """uint8 face crop -> the released agent's normalised 3-channel tensor (the extra channels are computed in the model)."""
    if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
        import cv2
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = ((np.ascontiguousarray(img).astype(np.float32) / 255.0) - 0.5) / 0.5
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


class XceptionChannelsDetector(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5, pretrained=True):
        super().__init__()
        self.channels = ForensicChannels()
        self.base_model = timm.create_model("xception", pretrained=pretrained)
        old = self.base_model.conv1
        conv1 = nn.Conv2d(3 + EXTRA, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
        with torch.no_grad():
            conv1.weight.zero_(); conv1.weight[:, :3] = old.weight  # starts as the RGB network; the new channels are learnt
        self.base_model.conv1 = conv1
        for p in self.base_model.parameters():
            p.requires_grad = False
        for p in self.base_model.conv1.parameters():
            p.requires_grad = True
        num_features = self.base_model.get_classifier().in_features
        self.attention = nn.Sequential(nn.Linear(num_features, num_features // 16), nn.ReLU(inplace=True),
                                       nn.Linear(num_features // 16, num_features), nn.Sigmoid())
        self.base_model.fc = nn.Sequential(nn.Linear(num_features, 1024), nn.ReLU(inplace=True), nn.BatchNorm1d(1024), nn.Dropout(dropout_rate),
                                           nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512), nn.Dropout(dropout_rate * 0.7),
                                           nn.Linear(512, num_classes))
        for p in list(self.base_model.fc.parameters()) + list(self.attention.parameters()):
            p.requires_grad = True

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            extra = self.channels(x.float() * 0.5 + 0.5)
        x = torch.cat([x, extra.to(x.dtype)], dim=1)
        f = self.base_model.global_pool(self.base_model.forward_features(x))
        return self.base_model.fc(f * self.attention(f))

    def unfreeze_from_block(self, block_num=11):
        unfreeze = False
        for name, module in self.base_model.named_modules():
            if f"block{block_num}" in name:
                unfreeze = True
            if unfreeze:
                for p in module.parameters():
                    p.requires_grad = True
        for p in list(self.base_model.fc.parameters()) + list(self.attention.parameters()) + list(self.base_model.conv1.parameters()):
            p.requires_grad = True
