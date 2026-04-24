
# Two-stage training script for a face quality model.

import argparse
import json
import logging
import os
import warnings
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                           recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

BASE_CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "data_dir": "data/polyglot_processed_all_unbalanced",
    "output_dir": "biometric_trained_models",
    "batch_size": 16,
    "num_workers": 4,
    "face_quality": { "image_size": 299 },
    "early_stopping": { "patience": 7, "min_delta": 0.0001 }
}

class PolyglotFakeDataset(Dataset):
    # Dataset to load .npz files and create 5-channel (RGB + quality) inputs.
    def __init__(self, data_dir: str, split: str = 'train', augment: bool = True,
                 normalize: bool = False, image_size: int = 299):
        self.data_dir = os.path.join(data_dir, split)
        self.files = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        self.augment = augment and (split == 'train')
        self.image_size = image_size
        self.transform = self._build_transforms(normalize)
        logging.info(f"Loaded {len(self.files)} files from {split} set.")

    def _build_transforms(self, normalize: bool) -> transforms.Compose:
        # Constructs the image transformation pipeline.
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
        ]
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        transform_list.append(transforms.ToTensor())
        if normalize:
            # Standard ImageNet normalization.
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = np.load(file_path, allow_pickle=True)
        faces = data['faces']
        label = 1 if data['label'][0] == 'fake' else 0
        face_quality_input = self._prepare_face_quality_data(faces)
        return {
            'face_quality': face_quality_input,
            'label': torch.tensor(label, dtype=torch.float32)
        }

    def _prepare_face_quality_data(self, faces: np.ndarray) -> torch.Tensor:
        # Prepares the 5-channel tensor (RGB + blur + exposure).
        if len(faces) == 0:
            return torch.zeros((5, self.image_size, self.image_size)) # Handle empty face arrays.
        
        face = faces[len(faces) // 2] # Use middle face.
        face_tensor = self.transform(face)
        quality_features = self._extract_quality_metrics(face)
        return torch.cat([face_tensor, quality_features], dim=0)

    def _extract_quality_metrics(self, face: np.ndarray) -> torch.Tensor:
        # Calculates blur and exposure scores.
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var() # Blur metric (Laplacian variance).
        blur_tensor = torch.full((1, self.image_size, self.image_size), blur_score / 1000.0)
        exposure_score = np.mean(gray) / 255.0 # Exposure metric (mean intensity).
        exposure_tensor = torch.full((1, self.image_size, self.image_size), exposure_score)
        return torch.cat([blur_tensor, exposure_tensor], dim=0)


class FaceQualityNet(nn.Module):
    # EfficientNet-B0 adapted for a 5-channel input.
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self._adapt_input_layer() # Adapt for 5 channels.

        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity() # Get features before classifier.

        # Custom classification head.
        self.quality_head = nn.Sequential(
            nn.Linear(num_features, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
            nn.Linear(256, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

        # Layer groups for discriminative learning rates.
        self.layer_groups = [
            self.backbone.features[:4], self.backbone.features[4:6], self.backbone.features[6:],
            self.quality_head, self.classifier
        ]

    def _adapt_input_layer(self):
        # Modifies the input conv layer for 5 channels.
        orig_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(5, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size, stride=orig_conv.stride,
            padding=orig_conv.padding, bias=False
        )
        with torch.no_grad():
            self.backbone.features[0][0].weight[:, :3] = orig_conv.weight # Copy RGB weights.
            self.backbone.features[0][0].weight[:, 3:] = orig_conv.weight[:, :2] * 0.1 # Init new channels.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        quality_features = self.quality_head(features)
        return torch.sigmoid(self.classifier(quality_features))

    def freeze_backbone(self):
        logging.info("Freezing backbone parameters.")
        for param in self.backbone.parameters(): param.requires_grad = False

    def unfreeze_backbone(self):
        logging.info("Unfreezing backbone parameters.")
        for param in self.backbone.parameters(): param.requires_grad = True


class Trainer:
    # Manages the training and validation loop.
    def __init__(self, model: FaceQualityNet, device: torch.device, config: Dict):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.criterion = nn.BCELoss()
        self.optimizer = self._setup_optimizer()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['num_epochs'])
        self.best_val_metric = -float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

    def _setup_optimizer(self) -> optim.Optimizer:
        # Configures optimizer, using discriminative LR if fine-tuning.
        if self.config.get('finetune', {}).get('discriminative_lr', False):
            return self._setup_discriminative_lr_optimizer()
        else:
            return optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=0.01)

    def _setup_discriminative_lr_optimizer(self) -> optim.Optimizer:
        # Sets up different learning rates for different layer groups.
        param_groups = []
        base_lr = self.config['learning_rate']
        multiplier = self.config['finetune']['backbone_lr_multiplier']
        lr_scales = [multiplier**3, multiplier**2, multiplier, 1.0, 1.0] # Smaller LR for earlier layers.
        logging.info("Setting up discriminative learning rates.")
        for i, (group, lr_scale) in enumerate(zip(self.model.layer_groups, lr_scales)):
            lr = base_lr * lr_scale
            param_groups.append({'params': group.parameters(), 'lr': lr})
            logging.info(f"  - Layer Group {i}: lr = {lr:.2e}")
        return optim.AdamW(param_groups, weight_decay=0.01)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        # Unfreeze backbone mid-training if configured.
        if self.config.get('finetune', {}).get('freeze_backbone', False):
            if epoch == self.config['finetune']['unfreeze_at_epoch']:
                self.model.unfreeze_backbone()
                self.optimizer = self._setup_optimizer() # Re-init optimizer with new params.

        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
        for batch in pbar:
            inputs = batch['face_quality'].to(self.device)
            labels = batch['label'].to(self.device).unsqueeze(1)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            running_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        return running_loss / len(dataloader), roc_auc_score(all_labels, all_preds)

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                inputs = batch['face_quality'].to(self.device)
                labels = batch['label'].to(self.device).unsqueeze(1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        preds_binary = (np.array(all_preds) > 0.5).astype(int)
        return {
            'loss': running_loss / len(dataloader),
            'auc': roc_auc_score(all_labels, all_preds),
            'accuracy': accuracy_score(all_labels, preds_binary),
            'f1': f1_score(all_labels, preds_binary, zero_division=0),
        }

    def check_early_stopping(self, val_metric: float) -> bool:
        patience = self.config['early_stopping']['patience']
        min_delta = self.config['early_stopping']['min_delta']
        if val_metric > self.best_val_metric + min_delta:
            self.best_val_metric = val_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            logging.info(f"Early stopping counter: {self.patience_counter}/{patience}")
            return self.patience_counter >= patience

    def save_checkpoint(self, epoch: int, path: str, val_metrics: Dict):
        checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                      'val_metrics': val_metrics, 'history': self.history}
        torch.save(checkpoint, path)


def run_training_stage(config: Dict, stage_name: str, base_output_dir: str,
                       pretrained_path: str = None) -> str:
    # Runs a full training stage (initial or fine-tuning).
    logging.info(f"\n{'='*25} STARTING STAGE: {stage_name.upper()} {'='*25}")
    experiment_dir = os.path.join(base_output_dir, stage_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save config for this stage.
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, default=str)

    normalize_data = (stage_name == "fine_tuning") # Normalize only for fine-tuning.
    train_dataset = PolyglotFakeDataset(config['data_dir'], 'train', augment=True, normalize=normalize_data, image_size=config['face_quality']['image_size'])
    val_dataset = PolyglotFakeDataset(config['data_dir'], 'val', augment=False, normalize=normalize_data, image_size=config['face_quality']['image_size'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)

    model = FaceQualityNet()
    if pretrained_path:
        logging.info(f"Loading pre-trained model from: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path)['model_state_dict'])
        if config.get('finetune', {}).get('freeze_backbone', False):
            model.freeze_backbone()

    trainer = Trainer(model, config['device'], config)
    best_val_auc = 0.0
    best_model_path = ""

    for epoch in range(1, config['num_epochs'] + 1):
        lr = trainer.scheduler.get_last_lr()[0]
        logging.info(f"\n--- Epoch {epoch}/{config['num_epochs']} | LR: {lr:.2e} ---")
        train_loss, train_auc = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader)
        trainer.history['train_loss'].append(train_loss)
        trainer.history['train_auc'].append(train_auc)
        trainer.history['val_loss'].append(val_metrics['loss'])
        trainer.history['val_auc'].append(val_metrics['auc'])
        logging.info(f"Train -> Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
        logging.info(f"Val   -> Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        trainer.scheduler.step()

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_path = os.path.join(experiment_dir, "best_model.pth")
            trainer.save_checkpoint(epoch, best_model_path, val_metrics)
            logging.info(f"New best model saved with Val AUC: {best_val_auc:.4f}")

        if trainer.check_early_stopping(val_metrics['auc']):
            logging.info(f"Early stopping triggered at epoch {epoch}.")
            break

    logging.info(f"\n{'='*25} STAGE '{stage_name.upper()}' COMPLETE {'='*25}")
    logging.info(f"Best model for this stage saved to: {best_model_path}")
    return best_model_path


def main():
    # Main function to run the two-stage training pipeline.
    parser = argparse.ArgumentParser(description='Automated Two-Stage Training for Face Quality Assessment.')
    parser.add_argument('--data_dir', type=str, default=BASE_CONFIG['data_dir'])
    parser.add_argument('--output_dir', type=str, default=BASE_CONFIG['output_dir'])
    parser.add_argument('--batch_size', type=int, default=BASE_CONFIG['batch_size'])
    parser.add_argument('--initial_epochs', type=int, default=50)
    parser.add_argument('--initial_lr', type=float, default=1e-3)
    parser.add_argument('--finetune_epochs', type=int, default=20)
    parser.add_argument('--finetune_lr', type=float, default=1e-5)
    args = parser.parse_args()

    # Stage 1: Initial training.
    stage1_config = {**BASE_CONFIG, "num_epochs": args.initial_epochs, "learning_rate": args.initial_lr}
    best_model_from_stage1 = run_training_stage(config=stage1_config, stage_name="initial_training", base_output_dir=args.output_dir)

    if not best_model_from_stage1:
        logging.error("Stage 1 failed. Halting.")
        return

    # Stage 2: Fine-tuning.
    stage2_config = {
        **BASE_CONFIG, "num_epochs": args.finetune_epochs, "learning_rate": args.finetune_lr,
        "finetune": {
            "freeze_backbone": True, "unfreeze_at_epoch": 5,
            "discriminative_lr": True, "backbone_lr_multiplier": 0.1,
        }
    }
    run_training_stage(config=stage2_config, stage_name="fine_tuning",
                       base_output_dir=args.output_dir, pretrained_path=best_model_from_stage1)
    
    logging.info("Automated training process completed.")

if __name__ == "__main__":
    main()