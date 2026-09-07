"""Visual agent variant: XceptionNet over the RGB face crop plus five forensic channels
computed from the same crop: error-level-analysis map, log-magnitude Fourier spectrum,
local-binary-pattern texture code, and the Cb and Cr chroma planes. The biometric-quality
agent already consumes a Laplacian sharpness map and a high-frequency residual map, so those
cues are deliberately left to it. Head, attention block and inference recipe (frame mean with
horizontal-flip test-time augmentation) are those of the released visual agent."""
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn

IMAGE_SIZE = 299
EXTRA = 5


def lbp_map(gray_u8):
    """8-neighbour local binary pattern code (radius 1) as a float map in [-1, 1]."""
    g = gray_u8.astype(np.int16); p = np.pad(g, 1, mode="edge"); h, w = g.shape
    code = np.zeros((h, w), np.int32)
    for bit, (dy, dx) in enumerate(((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))):
        code |= (p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] >= g).astype(np.int32) << bit
    return (code.astype(np.float32) / 127.5) - 1.0


def extra_channels(img):
    """uint8 HxWx3 face crop (stored channel order) -> float32 [5, H, W] in [-1, 1]."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    rec = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img
    ela = np.clip(np.abs(img.astype(np.float32) - rec.astype(np.float32)).mean(axis=2) / 255.0 * 16.0, 0.0, 1.0) * 2.0 - 1.0
    g = gray.astype(np.float32) / 255.0
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6) * 2.0 - 1.0
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
    cr = ycrcb[..., 1] * 2.0 - 1.0; cb = ycrcb[..., 2] * 2.0 - 1.0
    return np.stack([ela, mag.astype(np.float32), lbp_map(gray), cb, cr]).astype(np.float32)


def to_input(img):
    """uint8 face crop -> torch [3 + EXTRA, 299, 299]; RGB channels normalised as the released agent."""
    if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    img = np.ascontiguousarray(img)
    rgb = ((img.astype(np.float32) / 255.0) - 0.5) / 0.5
    return torch.from_numpy(np.concatenate([rgb.transpose(2, 0, 1), extra_channels(img)], axis=0))


class XceptionChannelsDetector(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5, pretrained=True):
        super().__init__()
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
