
"""
Trains the FreqNet audio-based deepfake detection model from preprocessed data.

This script incorporates modern training techniques, including:
- On-the-fly spectrogram augmentations using torchaudio.
- The AdamW optimizer for improved weight decay handling.
- The OneCycleLR learning rate scheduler for faster and more stable convergence.
- Early stopping to prevent overfitting.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from sklearn.metrics import accuracy_score
import glob
import warnings
import matplotlib.pyplot as plt

# Attempt to import torchaudio for augmentations.
try:
    import torchaudio
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("Warning: torchaudio not found. Audio augmentations will be skipped. Install with 'pip install torchaudio'")

# Ignore unnecessary warnings for a cleaner output.
warnings.filterwarnings('ignore', category=UserWarning)

def get_preprocessed_splits(preprocessed_path: str) -> tuple[list, list]:
    """
    Finds and returns the file paths for the preprocessed train and validation sets.
    """
    train_dir = os.path.join(preprocessed_path, 'train')
    val_dir = os.path.join(preprocessed_path, 'val')

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Preprocessed directories not found. Please run the data preparation script first.")

    print(f"--- Loading data splits from '{preprocessed_path}' ---")
    train_files = glob.glob(os.path.join(train_dir, '*.npz'))
    val_files = glob.glob(os.path.join(val_dir, '*.npz'))

    if not train_files or not val_files:
        raise FileNotFoundError(f"No .npz files found in {train_dir} or {val_dir}. Check if preprocessing ran correctly.")

    print(f"Found {len(train_files)} training files and {len(val_files)} validation files.")
    return train_files, val_files

def get_options():
    """Parses command-line arguments for training."""
    parser = argparse.ArgumentParser(description="FreqNet Audio Model Training Script")
    parser.add_argument('--dataroot', type=str, default='polyglot_processed_all_unbalanced', help='Path to preprocessed data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Input batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Maximum learning rate for OneCycleLR')
    parser.add_argument('--patience', type=int, default=10, help='Epochs to wait for improvement before early stopping')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
    parser.add_argument('--n_mels', type=int, default=224, help='Number of Mel bands for spectrograms')
    parser.add_argument('--output_model_path', type=str, default='freqnet_model_all_unbalanced_improved.pth', help='Path to save the best model')
    parser.add_argument('--plot_path', type=str, default='training_curves_freqnet_improved.png', help='Path to save training curves plot')
    parser.add_argument('--device', type=str, default='auto', help='Device: "auto", "cuda", or "cpu"')
    args = parser.parse_args()
    if args.device == 'auto': args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("-" * 20, "\nTraining Options:"); [print(f"  - {k}: {v}") for k, v in vars(args).items()]; print("-" * 20)
    return args

def plot_curves(train_loss_hist, val_acc_hist, save_path):
    """Plots and saves the training loss and validation accuracy curves."""
    epochs = range(1, len(train_loss_hist) + 1)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Training and Validation Metrics', fontsize=16)

    ax1.plot(epochs, train_loss_hist, 'o-', color='dodgerblue', label='Training Loss')
    ax1.set_ylabel('Loss'); ax1.set_title('Training Loss per Epoch'); ax1.legend()

    ax2.plot(epochs, val_acc_hist, 'o-', color='crimson', label='Validation Accuracy')
    ax2.set_ylabel('Accuracy'); ax2.set_xlabel('Epochs')
    ax2.set_title('Validation Accuracy per Epoch'); ax2.set_ylim(0, 1.05); ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"\nTraining curves plot saved to {save_path}")
    plt.close()

class PreprocessedAudioDataset(Dataset):
    """
    A PyTorch dataset to load preprocessed audio from .npz files,
    generate Mel spectrograms, and apply augmentations on the fly.
    """
    def __init__(self, file_list, sample_rate, n_mels, is_train=False):
        self.file_paths = file_list
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.is_train = is_train

        # Initialize augmentations for the training set if torchaudio is available.
        if self.is_train and TORCHAUDIO_AVAILABLE:
            self.augmentation = nn.Sequential(
                T.FrequencyMasking(freq_mask_param=30), # Masks out a range of frequencies.
                T.TimeMasking(time_mask_param=50)      # Masks out a range of time steps.
            )
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        try:
            data = np.load(filepath, allow_pickle=True)
            waveform = data['waveform']
            label = 1.0 if data['label'][0] == 'fake' else 0.0
            if waveform.size == 0: return None

            # Lazily import librosa to avoid issues if it's not installed system-wide.
            import librosa
            
            # Pad or truncate the audio to a fixed length (5 seconds).
            target_length = 5 * self.sample_rate
            waveform = waveform[:target_length] if len(waveform) > target_length else np.pad(waveform, (0, target_length - len(waveform)))

            mel_spec = librosa.feature.melspectrogram(y=waveform, sr=self.sample_rate, n_mels=self.n_mels)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            normalized_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-6)
            
            # Prepare tensor for a 2D CNN (C, H, W) - repeat mono spec across 3 channels.
            spec_tensor = torch.tensor(normalized_spec, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

            # Apply augmentations during training.
            if self.augmentation:
                spec_tensor = self.augmentation(spec_tensor)

        except Exception as e:
            print(f"Warning: Could not process {os.path.basename(filepath)}. Skipping. Error: {e}")
            return None

        return spec_tensor, torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    """A custom collate function to filter out None values from the batch."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None, None
    return torch.utils.data.dataloader.default_collate(batch)

# --- FreqNet Model Definition ---
# Helper functions and classes for building the ResNet-like architecture.
def conv3x3(in_planes, out_planes, stride=1): return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1): return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes); self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride); self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion); self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True); self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return self.relu(out)

class FreqNet(nn.Module):
    """
    FreqNet: A ResNet-based architecture that processes spectrograms using
    custom frequency-domain operations involving Fast Fourier Transforms (FFTs).
    """
    def __init__(self, block=Bottleneck, layers=[3, 4], num_classes=1):
        super(FreqNet, self).__init__()
        self.weight1 = nn.Parameter(torch.randn(64, 3, 1, 1)); self.bias1 = nn.Parameter(torch.randn(64))
        self.realconv1 = conv1x1(64, 64); self.imagconv1 = conv1x1(64, 64)
        self.weight2 = nn.Parameter(torch.randn(64, 64, 1, 1)); self.bias2 = nn.Parameter(torch.randn(64))
        self.realconv2 = conv1x1(64, 64); self.imagconv2 = conv1x1(64, 64)
        self.weight3 = nn.Parameter(torch.randn(256, 256, 1, 1)); self.bias3 = nn.Parameter(torch.randn(256))
        self.realconv3 = conv1x1(256, 256); self.imagconv3 = conv1x1(256, 256)
        self.weight4 = nn.Parameter(torch.randn(256, 256, 1, 1)); self.bias4 = nn.Parameter(torch.randn(256))
        self.realconv4 = conv1x1(256, 256); self.imagconv4 = conv1x1(256, 256)
        
        self.inplanes = 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Add a dropout layer for regularization before the final linear layer.
        self.fc1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes))
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks): layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _fft_conv(self, x, weight, bias, real_conv, imag_conv):
        """Helper for a single FFT-based convolution block."""
        x = F.conv2d(x, weight, bias, stride=1, padding=0)
        x = F.relu(x, inplace=True)
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.complex(real_conv(x.real), imag_conv(x.imag))
        x = torch.fft.ifft2(x, norm="ortho")
        return F.relu(torch.real(x), inplace=True)

    def forward(self, x):
        # The forward pass consists of several blocks that apply FFTs, convolutions,
        # and standard ResNet layers to process the spectrogram.
        x = self._fft_conv(x, self.weight1, self.bias1, self.realconv1, self.imagconv1)
        x = self._fft_conv(x, self.weight2, self.bias2, self.realconv2, self.imagconv2)
        
        x = self.maxpool(x)
        x = self.layer1(x)
        
        x = self._fft_conv(x, self.weight3, self.bias3, self.realconv3, self.imagconv3)
        x = self._fft_conv(x, self.weight4, self.bias4, self.realconv4, self.imagconv4)

        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x.squeeze(1)

def main(opt):
    """The main function to run the training and validation pipeline."""
    train_files, val_files = get_preprocessed_splits(opt.dataroot)

    train_dataset = PreprocessedAudioDataset(train_files, opt.sample_rate, opt.n_mels, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    val_dataset = PreprocessedAudioDataset(val_files, opt.sample_rate, opt.n_mels)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    model = FreqNet().to(opt.device)
    criterion = nn.BCEWithLogitsLoss()
    # Use AdamW, an improved version of Adam that decouples weight decay from the optimization step.
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    # Use OneCycleLR, which warms up the learning rate, reaches a max, and then anneals down.
    scheduler = OneCycleLR(optimizer, max_lr=opt.lr, steps_per_epoch=len(train_loader), epochs=opt.epochs)

    print(f"\nStarting training on device: {opt.device}")

    best_val_accuracy = 0.0
    epochs_no_improve = 0
    train_loss_history, val_accuracy_history = [], []

    for epoch in range(opt.epochs):
        model.train()
        total_train_loss = 0.0
        for specs, labels in train_loader:
            if specs is None: continue
            specs, labels = specs.to(opt.device), labels.to(opt.device)
            
            outputs = model(specs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step() # The scheduler is stepped after each batch.

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        model.eval()
        all_preds, all_labels = [], []
        val_loss_total = 0.0
        with torch.no_grad():
            for specs, labels in val_loader:
                if specs is None: continue
                specs, labels = specs.to(opt.device), labels.to(opt.device)
                outputs = model(specs)
                val_loss_total += criterion(outputs, labels).item()
                preds = (outputs > 0.0).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0

        train_loss_history.append(avg_train_loss)
        val_accuracy_history.append(val_accuracy)

        print("-" * 50)
        print(f"End of Epoch {epoch+1}/{opt.epochs} | Avg Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss_total/max(len(val_loader),1):.4f} | Val Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), opt.output_model_path)
            print(f"New best validation accuracy: {best_val_accuracy:.4f}. Model saved to {opt.output_model_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= opt.patience:
            print(f"--- Early stopping triggered after {opt.patience} epochs with no improvement. ---")
            break
        print("-" * 50)

    print(f"\nTraining complete. Best model saved to {opt.output_model_path} with validation accuracy: {best_val_accuracy:.4f}")

    if train_loss_history:
        plot_curves(train_loss_history, val_accuracy_history, opt.plot_path)


if __name__ == '__main__':
    options = get_options()
    main(options)