import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import glob
from collections import defaultdict
import re
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import warnings

# Install timm if not available: pip install timm
try:
    import timm
except ImportError:
    print("Please install timm: pip install timm")
    raise

# Install Albumentations if not available: pip install albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("Please install Albumentations: pip install albumentations")
    raise

warnings.filterwarnings('ignore')

# --- Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- Path Configuration ---
BASE_DATA_DIR = Path("polyglot_processed_all_unbalanced")
TRAIN_DATA_PATH = BASE_DATA_DIR / "train"
VAL_DATA_PATH = BASE_DATA_DIR / "val"
OUTPUT_DIR = Path("tuning_and_model_output_unbal_all_face_cutout")

# Define paths for all save points
INITIAL_TRAINED_MODEL_PATH = OUTPUT_DIR / "initial_trained_model_pytorch_xception_unbal.pth"
BEST_MODEL_PATH = OUTPUT_DIR / "polyglotfake_xception_best_pytorch_unbal_all.pth"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint_epoch_{}.pth"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Model & Training Hyperparameters ---
IMAGE_SIZE = 299
BATCH_SIZE = 32 if torch.cuda.is_available() else 8
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 20  # Increased for better convergence
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1  # Slightly increased
MIXUP_ALPHA = 0.2  # NEW: Mixup augmentation parameter

# Fine-tuning configuration
FINE_TUNE_AT_BLOCK = 11  # Unfreeze slightly earlier


class FaceCutout(A.ImageOnlyTransform):
    """Custom Face-aware Cutout augmentation that focuses on facial regions"""
    
    def __init__(self, num_holes=1, max_h_size=40, max_w_size=40, 
                 face_focus_prob=0.7, fill_value=0, always_apply=False, p=1.0):
        super(FaceCutout, self).__init__(always_apply, p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.face_focus_prob = face_focus_prob
        self.fill_value = fill_value
    
    def apply(self, image, **params):
        h, w = image.shape[:2]
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.num_holes):
            # Size of the hole
            hole_h = np.random.randint(1, self.max_h_size)
            hole_w = np.random.randint(1, self.max_w_size)
            
            if np.random.random() < self.face_focus_prob:
                # Focus on face regions (center-biased)
                # Faces are usually in the center 60% of the image
                center_h, center_w = h // 2, w // 2
                y1 = np.random.randint(
                    max(0, center_h - h // 4),
                    min(h, center_h + h // 4) - hole_h
                )
                x1 = np.random.randint(
                    max(0, center_w - w // 4),
                    min(w, center_w + w // 4) - hole_w
                )
            else:
                # Random position
                y1 = np.random.randint(0, h - hole_h)
                x1 = np.random.randint(0, w - hole_w)
            
            y2 = y1 + hole_h
            x2 = x1 + hole_w
            
            mask[y1:y2, x1:x2] = 0
        
        # Apply mask
        image = image * mask[:, :, np.newaxis] + self.fill_value * (1 - mask[:, :, np.newaxis])
        
        return image.astype(image.dtype)
    
    def get_transform_init_args_names(self):
        return ("num_holes", "max_h_size", "max_w_size", "face_focus_prob", "fill_value")


class FaceDataset(Dataset):
    """PyTorch Dataset for face data with enhanced augmentation support"""
    
    def __init__(self, file_paths: List[Path], transform=None, mixup=False):
        self.file_paths = file_paths
        self.transform = transform
        self.mixup = mixup
        self.face_map: List[Tuple[Path, int]] = []
        self.labels = []
        
        logging.info(f"Building face map from {len(file_paths)} data files...")
        
        files_processed = 0
        total_faces = 0
        
        for i, path in enumerate(file_paths):
            try:
                with np.load(path, allow_pickle=True) as data:
                    # Debug: print keys for first file
                    if i == 0:
                        logging.info(f"First file keys: {list(data.keys())}")
                    
                    # Try different possible key names
                    faces_key = None
                    for key in ['faces', 'face', 'images', 'image', 'data']:
                        if key in data:
                            faces_key = key
                            break
                    
                    if faces_key is None:
                        logging.warning(f"No face data found in {path.name}, keys: {list(data.keys())}")
                        continue
                    
                    faces = data[faces_key]
                    num_faces = len(faces) if hasattr(faces, '__len__') else 1
                    
                    if num_faces > 0:
                        self.face_map.extend([(path, i) for i in range(num_faces)])
                        
                        # Handle different label formats
                        if "label" in data:
                            label_data = data["label"]
                            if isinstance(label_data, (list, np.ndarray)):
                                label = 1 if str(label_data[0]).lower() == "fake" else 0
                            else:
                                label = 1 if str(label_data).lower() == "fake" else 0
                        elif "labels" in data:
                            label_data = data["labels"]
                            if isinstance(label_data, (list, np.ndarray)):
                                label = 1 if str(label_data[0]).lower() == "fake" else 0
                            else:
                                label = 1 if str(label_data).lower() == "fake" else 0
                        else:
                            # Try to infer from filename
                            label = 1 if "fake" in path.name.lower() else 0
                            
                        self.labels.extend([label] * num_faces)
                        files_processed += 1
                        total_faces += num_faces
                        
            except Exception as e:
                logging.error(f"Could not read {path.name}: {e}")
        
        self.num_samples = len(self.face_map)
        logging.info(f"Successfully processed {files_processed}/{len(self.file_paths)} files")
        logging.info(f"Found {self.num_samples} faces total")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        path, face_idx = self.face_map[idx]
        
        with np.load(path, allow_pickle=True) as data:
            # Find the faces key
            faces_key = None
            for key in ['faces', 'face', 'images', 'image', 'data']:
                if key in data:
                    faces_key = key
                    break
            
            if faces_key is None:
                raise KeyError(f"No face data found in {path.name}")
            
            faces = data[faces_key]
            
            # Handle single face vs multiple faces
            if hasattr(faces, '__len__') and len(faces) > face_idx:
                face = faces[face_idx]
            else:
                face = faces if face_idx == 0 else faces
            
            # Handle label
            if "label" in data:
                label_data = data["label"]
                if isinstance(label_data, (list, np.ndarray)):
                    label = 1 if str(label_data[0]).lower() == "fake" else 0
                else:
                    label = 1 if str(label_data).lower() == "fake" else 0
            elif "labels" in data:
                label_data = data["labels"]
                if isinstance(label_data, (list, np.ndarray)):
                    label = 1 if str(label_data[0]).lower() == "fake" else 0
                else:
                    label = 1 if str(label_data).lower() == "fake" else 0
            else:
                label = 1 if "fake" in path.name.lower() else 0
            
            # Ensure face is 3D (H, W, C)
            if len(face.shape) == 2:  # Grayscale
                face = np.stack([face, face, face], axis=-1)
            elif len(face.shape) == 3 and face.shape[0] == 3:  # (C, H, W)
                face = np.transpose(face, (1, 2, 0))
            
            # Convert to uint8 for albumentations
            if face.dtype != np.uint8:
                if face.max() <= 1.0:
                    face = (face * 255).astype(np.uint8)
                else:
                    face = face.astype(np.uint8)
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=face)
                face = transformed['image']
            
            return face, label


class XceptionDeepfakeDetector(nn.Module):
    """Enhanced Xception detector with attention mechanism"""
    
    def __init__(self, num_classes=1, dropout_rate=0.5, pretrained=True):
        super(XceptionDeepfakeDetector, self).__init__()
        
        self.base_model = timm.create_model('xception', pretrained=pretrained)
        
        # Freeze all layers initially
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        num_features = self.base_model.get_classifier().in_features
        
        # Enhanced classifier head with attention
        self.attention = nn.Sequential(
            nn.Linear(num_features, num_features // 16),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // 16, num_features),
            nn.Sigmoid()
        )
        
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, num_classes)
        )
        
        # Make sure the new classifier and attention are trainable
        for param in self.base_model.fc.parameters():
            param.requires_grad = True
        for param in self.attention.parameters():
            param.requires_grad = True
        
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
    
    def forward(self, x):
        # Extract features before the final classifier
        features = self.base_model.forward_features(x)
        features = self.base_model.global_pool(features)
        
        # Apply attention
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Final classification
        return self.base_model.fc(features)
    
    def unfreeze_from_block(self, block_num=11):
        unfreeze = False
        for name, module in self.base_model.named_modules():
            if f'block{block_num}' in name:
                unfreeze = True
            if unfreeze:
                for param in module.parameters():
                    param.requires_grad = True
        
        for param in self.base_model.fc.parameters():
            param.requires_grad = True
        for param in self.attention.parameters():
            param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Unfroze layers from block{block_num}")
        logging.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")


def get_transforms():
    """Enhanced data augmentation with Face Cutout"""
    
    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE + 20, IMAGE_SIZE + 20),  # Resize slightly larger
        A.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),  # Random crop to target size
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1),
        ], p=0.8),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MedianBlur(blur_limit=5, p=1),
        ], p=0.5),
        FaceCutout(
            num_holes=2,
            max_h_size=50,
            max_w_size=50,
            face_focus_prob=0.8,
            fill_value=0,
            p=0.5
        ),  # Apply Face Cutout
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.15, 
            rotate_limit=15, 
            border_mode=0,
            p=0.6
        ),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1),
        ], p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.4),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, 
                class_weights=None, scheduler=None, use_mixup=True):
    """Train for one epoch with mixup and advanced techniques"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        original_labels = labels.float().to(device)
        
        # Apply Mixup
        if use_mixup and np.random.random() > 0.5:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, original_labels, MIXUP_ALPHA)
            
            # Apply label smoothing to both labels
            smoothed_labels_a = labels_a * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            smoothed_labels_b = labels_b * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            
            # Mixup loss
            loss_a = criterion(outputs, smoothed_labels_a)
            loss_b = criterion(outputs, smoothed_labels_b)
            
            if class_weights is not None:
                weights_a = torch.where(labels_a == 1, class_weights[1], class_weights[0])
                weights_b = torch.where(labels_b == 1, class_weights[1], class_weights[0])
                loss_a = (loss_a * weights_a).mean()
                loss_b = (loss_b * weights_b).mean()
            
            loss = lam * loss_a + (1 - lam) * loss_b
            
        else:
            # Standard training with label smoothing
            smoothed_labels = original_labels * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, smoothed_labels)
            
            if class_weights is not None:
                weights = torch.where(original_labels == 1, class_weights[1], class_weights[0])
                loss = (loss * weights).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        running_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0).float()
        total += original_labels.size(0)
        correct += (predicted == original_labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, class_weights=None):
    """Validate with Test Time Augmentation (TTA)"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # TTA settings
    tta_transforms = A.Compose([
        A.HorizontalFlip(p=1.0),
    ])
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            

            outputs = model(inputs).squeeze()

            
            loss = criterion(outputs, labels)
            
            if class_weights is not None:
                weights = torch.where(labels == 1, class_weights[1], class_weights[0])
                loss = (loss * weights).mean()
            
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def main():
    """Main training function with enhanced features"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Enable mixed precision training if available
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Check if data directories exist
    if not TRAIN_DATA_PATH.exists():
        logging.error(f"Train data path does not exist: {TRAIN_DATA_PATH}")
        logging.info("Please check your data directory structure")
        return
    
    if not VAL_DATA_PATH.exists():
        logging.error(f"Validation data path does not exist: {VAL_DATA_PATH}")
        logging.info("Please check your data directory structure")
        return
    

    train_files = list(TRAIN_DATA_PATH.rglob("*.npz"))
    val_files = list(VAL_DATA_PATH.rglob("*.npz"))
    
    logging.info(f"Found {len(train_files)} training files")
    logging.info(f"Found {len(val_files)} validation files")
    
    if len(train_files) == 0:
        logging.error("No training files found!")
        logging.info(f"Looking for .npz files in: {TRAIN_DATA_PATH}")

        all_files = list(TRAIN_DATA_PATH.rglob("*"))
        if all_files:
            logging.info(f"Found {len(all_files)} files total, showing first 5:")
            for f in all_files[:5]:
                logging.info(f"  - {f.name} ({f.suffix})")
        return
    
    if len(val_files) == 0:
        logging.error("No validation files found!")
        logging.info(f"Looking for .npz files in: {VAL_DATA_PATH}")
        return
    
    train_transform, val_transform = get_transforms()
    train_dataset = FaceDataset(train_files, transform=train_transform, mixup=True)
    val_dataset = FaceDataset(val_files, transform=val_transform, mixup=False)
    
    # Check if datasets have samples
    if len(train_dataset) == 0:
        logging.error("Training dataset is empty! No faces found in training files.")
        return
    
    if len(val_dataset) == 0:
        logging.error("Validation dataset is empty! No faces found in validation files.")
        return
    
    # Compute class weights only if labels
    if len(train_dataset.labels) > 0:
        unique_classes = np.unique(train_dataset.labels)
        if len(unique_classes) < 2:
            logging.warning(f"Only one class found in training data: {unique_classes}")
            logging.warning("Using uniform class weights")
            class_weights = torch.FloatTensor([1.0, 1.0]).to(device)
        else:
            class_weights_val = compute_class_weight(
                "balanced", 
                classes=unique_classes, 
                y=train_dataset.labels
            )
            class_weights = torch.FloatTensor(class_weights_val).to(device)
    else:
        logging.error("No labels found in training dataset!")
        return
    logging.info(f"Class weights: {class_weights.cpu().numpy()}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True  
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    logging.info("\nInitializing Enhanced Xception model...")
    model = XceptionDeepfakeDetector(num_classes=1, dropout_rate=0.5, pretrained=True).to(device)
    

    criterion = nn.BCEWithLogitsLoss(reduction='none')

    
    start_epoch = 0
    best_val_acc = 0.0
    

    if BEST_MODEL_PATH.exists():
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        logging.info(f"Resumed from epoch {start_epoch} with best val acc: {best_val_acc:.4f}")
    
    # Initial training with frozen base
    if start_epoch < INITIAL_EPOCHS:
        logging.info("\n--- PHASE 1: Initial Training (Frozen Base) ---")
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(
            trainable_params, 
            lr=INITIAL_LR, 
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        scheduler_plateau = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2, 
            verbose=True,
            min_lr=1e-7
        )
        
        for epoch in range(start_epoch + 1, INITIAL_EPOCHS + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, 
                epoch, class_weights, use_mixup=False
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device, class_weights)
            scheduler_plateau.step(val_loss)
            
            logging.info(
                f"Epoch {epoch}/{INITIAL_EPOCHS}: "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_acc': best_val_acc
                }, INITIAL_TRAINED_MODEL_PATH)
                logging.info(f"Saved initial model with validation accuracy: {val_acc:.4f}")
        
        start_epoch = INITIAL_EPOCHS
    
    # Phase 2: Fine-tuning with gradual unfreezing
    if start_epoch < TOTAL_EPOCHS:
        logging.info("\n--- PHASE 2: Fine-tuning with Gradual Unfreezing ---")
        
        model.unfreeze_from_block(FINE_TUNE_AT_BLOCK)
        
        # Differential learning rates
        base_params = []
        classifier_params = []
        attention_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'fc' in name:
                    classifier_params.append(param)
                elif 'attention' in name:
                    attention_params.append(param)
                else:
                    base_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': base_params, 'lr': FINE_TUNE_LR},
            {'params': attention_params, 'lr': FINE_TUNE_LR * 5},
            {'params': classifier_params, 'lr': FINE_TUNE_LR * 10}
        ], weight_decay=WEIGHT_DECAY)
        
        # Use OneCycleLR for fine-tuning
        steps_per_epoch = len(train_loader)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[FINE_TUNE_LR, FINE_TUNE_LR * 5, FINE_TUNE_LR * 10],
            total_steps=(TOTAL_EPOCHS - INITIAL_EPOCHS) * steps_per_epoch,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        patience_counter = 0
        best_fine_tune_acc = best_val_acc
        
        for epoch in range(start_epoch + 1, TOTAL_EPOCHS + 1):
            # Gradual unfreezing (optional)
            if epoch == start_epoch + 5:
                model.unfreeze_from_block(FINE_TUNE_AT_BLOCK - 1)
                logging.info(f"Further unfreezing at epoch {epoch}")
            
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device,
                epoch, class_weights, scheduler, use_mixup=True
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device, class_weights)
            
            logging.info(
                f"Epoch {epoch}/{TOTAL_EPOCHS}: "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )
            
            # Save checkpoint
            if epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_fine_tune_acc
                }, str(CHECKPOINT_PATH).format(epoch))
            
            if val_acc > best_fine_tune_acc:
                best_fine_tune_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_acc': best_fine_tune_acc
                }, BEST_MODEL_PATH)
                logging.info(f"Saved best model with validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= 7:  
                logging.info("Early stopping triggered after 7 epochs without improvement")
                break
    
    logging.info("\n--- Training Complete! ---")
    logging.info(f"Best validation accuracy achieved: {best_fine_tune_acc:.4f}")


if __name__ == "__main__":
    main()