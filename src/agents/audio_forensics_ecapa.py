# -*- coding: utf-8 -*-
"""
An optimized training script for an ECAPA-TDNN and audio feature model
"""

import os
import argparse
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import librosa
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from speechbrain.pretrained import EncoderClassifier

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "data_dir": "data/polyglot_processed_all_unbalanced",
    "output_dir": "audio_forensic_trained_models_v2",
    "batch_size": 16,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    # Forked DataLoader workers segfault when the SpeechBrain ECAPA backbone is already
    # resident in the parent: torchaudio/SpeechBrain hold non-fork-safe state, and both 4
    # and 2 workers died on the first batch. 0 keeps loading in-process. This stage is not
    # dataloader-bound - the ECAPA backbone is frozen and only a 43k-parameter head trains -
    # so the cost is negligible.
    "num_workers": 0,
    "early_stopping_patience": 7,
    "audio": {
        "sample_rate": 16000,
        "duration": 6,
        "embedding_dim": 192,
        "window_size": 2.0,
    },
    "model_params": {
        "embedding_dim": 192,
        "num_forensic_features": 11 # prosody(5) + artifacts(3) + temporal(2) + variance(1)
    }
}

# ==================== FEATURE EXTRACTION ====================
class FastAudioFeatureExtractor:
    """Extracts high-quality audio forensic features on the CPU."""
    @staticmethod
    def extract_fast_prosody(waveform, sr):
        f0, _, _ = librosa.pyin(waveform, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        pitch_mean, pitch_std, pitch_range = (np.mean(f0_clean), np.std(f0_clean), np.ptp(f0_clean)) if len(f0_clean) > 0 else (0.0, 0.0, 0.0)
        rms = librosa.feature.rms(y=waveform)
        energy, energy_std = np.mean(rms), np.std(rms)
        return np.array([pitch_mean, pitch_std, pitch_range, energy, energy_std], dtype=np.float32)

    @staticmethod
    def extract_fast_artifacts(waveform, sr):
        stft = np.abs(librosa.stft(waveform, n_fft=512))
        spectral_flux = np.mean(np.diff(stft.sum(axis=0))**2)
        # Spectral centroid often correlates with the "brightness" of a sound.
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sr, n_fft=512))
        high_freq_energy = np.mean(stft[stft.shape[0]//2:, :])
        low_freq_energy = np.mean(stft[:stft.shape[0]//2, :])
        hf_ratio = high_freq_energy / (low_freq_energy + 1e-8)
        return np.array([hf_ratio, spectral_flux, spectral_centroid], dtype=np.float32)

# ==================== DATASET ====================
class OptimizedAudioDataset(Dataset):
    """
    A Dataset that extracts speaker embeddings and forensic features on the CPU
    and applies pre-computed normalization statistics.
    """
    def __init__(self, data_dir: str, split: str = 'train', stats: dict = None):
        self.data_dir = os.path.join(data_dir, split)
        self.files = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )
        self.speaker_encoder.eval()
        self.feature_extractor = FastAudioFeatureExtractor()
        self.stats = stats
        print(f"Loaded {len(self.files)} files from {split} set.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        try:
            data = np.load(file_path, allow_pickle=True)
            waveform = data['waveform'].astype(np.float32)
            label = 1 if data['label'][0] == 'fake' else 0
        except Exception as e:
            print(f"Warning: Could not process {file_path}. Skipping. Error: {e}")
            dummy_dim = CONFIG['model_params']['embedding_dim'] + CONFIG['model_params']['num_forensic_features']
            return {'features': torch.zeros(dummy_dim), 'label': torch.tensor(0.0)}

        features = self._extract_features(waveform)

        if self.stats:
            features = (features - self.stats['mean']) / (self.stats['std'] + 1e-6)

        return {'features': features, 'label': torch.tensor(label, dtype=torch.float32)}

    def _extract_features(self, waveform):
        sr = CONFIG['audio']['sample_rate']
        if waveform.size < 400:
            dummy_dim = CONFIG['model_params']['embedding_dim'] + CONFIG['model_params']['num_forensic_features']
            return torch.zeros(dummy_dim)

        target_length = sr * CONFIG['audio']['duration']
        waveform = waveform[:target_length] if len(waveform) > target_length else np.pad(waveform, (0, target_length - len(waveform)))
        waveform_tensor = torch.tensor(waveform).unsqueeze(0)

        with torch.no_grad():
            embedding = self.speaker_encoder.encode_batch(waveform_tensor).squeeze().numpy()

        window_size = int(CONFIG['audio']['window_size'] * sr)
        window_positions = [0, len(waveform)//2 - window_size//2, len(waveform) - window_size]
        embeddings = [self.speaker_encoder.encode_batch(torch.tensor(waveform[p:p+window_size]).unsqueeze(0)).squeeze().numpy() for p in window_positions if p >= 0 and p+window_size <= len(waveform)]
        
        embeddings_arr = np.array(embeddings)
        if len(embeddings_arr) >= 2:
            distances = np.linalg.norm(embeddings_arr[:-1] - embeddings_arr[1:], axis=1)
            temporal_features = np.array([np.mean(distances), np.std(distances)])
            embedding_var = np.std(embeddings_arr, axis=0).mean()
        else:
            temporal_features = np.zeros(2)
            embedding_var = 0.0

        prosody_features = self.feature_extractor.extract_fast_prosody(waveform, sr)
        artifact_features = self.feature_extractor.extract_fast_artifacts(waveform, sr)
        
        all_features = np.concatenate([embedding, prosody_features, artifact_features, temporal_features, [embedding_var]])
        return torch.tensor(all_features, dtype=torch.float32)

# ==================== MODEL ====================
class OptimizedLightweightForensics(nn.Module):
    """
    A lightweight model that processes speaker embeddings and forensic features
    through separate streams before combining them for classification.
    """
    def __init__(self, embedding_dim, num_forensic_features):
        super().__init__()
        # Stream for processing the deep speaker embedding.
        self.embedding_processor = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        # Stream for processing the hand-crafted forensic features.
        self.forensic_processor = nn.Sequential(
            nn.Linear(num_forensic_features, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        # Final classifier head.
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(64, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        embedding_data = x[:, :CONFIG['model_params']['embedding_dim']]
        forensic_data = x[:, CONFIG['model_params']['embedding_dim']:]
        emb_features = self.embedding_processor(embedding_data)
        forensic_features = self.forensic_processor(forensic_data)
        combined = torch.cat([emb_features, forensic_features], dim=1)
        return self.classifier(combined)

# ==================== UTILITIES ====================
class EarlyStopping:
    """
    Stops training when a monitored metric has stopped improving.
    Saves the best model state based on this metric.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.inf # For loss-based monitoring
        self.delta = delta
        self.path = path
        
    def __call__(self, val_score, model):
        # We monitor AUC, so a higher score is better.
        if self.best_score is None or val_score > self.best_score + self.delta:
            self.best_score = val_score
            self.save_checkpoint(val_score, model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_score, model):
        if self.verbose:
            print(f'Validation metric improved. Saving model to {self.path}...')
        torch.save(model.state_dict(), self.path)

# ==================== MAIN TRAINING SCRIPT ====================
def main(args):
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    device = CONFIG['device']
    
    # Initialize GradScaler for automatic mixed-precision training.
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    print("Calculating training set statistics for normalization...")
    temp_train_dataset = OptimizedAudioDataset(CONFIG['data_dir'], 'train')
    temp_loader = DataLoader(temp_train_dataset, batch_size=args.batch_size, num_workers=CONFIG['num_workers'])
    
    all_features = torch.cat([b['features'] for b in tqdm(temp_loader, desc="Computing Stats")], dim=0)
    mean, std = torch.mean(all_features, dim=0), torch.std(all_features, dim=0)
    STATS = {'mean': mean, 'std': std}
    print("Statistics calculated.")
    
    stats_path = os.path.join(CONFIG['output_dir'], 'training_stats.npz')
    np.savez(stats_path, mean=mean.numpy(), std=std.numpy())
    print(f"Training statistics saved to {stats_path}")

    train_dataset = OptimizedAudioDataset(CONFIG['data_dir'], 'train', stats=STATS)
    val_dataset = OptimizedAudioDataset(CONFIG['data_dir'], 'val', stats=STATS)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)

    model = OptimizedLightweightForensics(**CONFIG['model_params']).to(device)
    print(f"\nModel initialized on device: {device}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG['learning_rate'] * 10, epochs=CONFIG['num_epochs'], steps_per_epoch=len(train_loader))

    best_model_path = os.path.join(CONFIG['output_dir'], "audio_forensics_model_best.pth")
    early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'], verbose=True, path=best_model_path)
    
    log_file = os.path.join(CONFIG['output_dir'], 'training_log.csv')
    log_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_auc', 'val_loss', 'val_auc', 'val_acc', 'lr'])

    for epoch in range(CONFIG['num_epochs']):
        print(f"\n{'='*50}\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        model.train()
        train_loss, train_labels, train_preds = 0.0, [], []
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            features = batch['features'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(features).squeeze(1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})

        model.eval()
        val_loss, val_labels, val_preds = 0.0, [], []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch in pbar:
                features, labels = batch['features'].to(device, non_blocking=True), batch['label'].to(device, non_blocking=True)
                with autocast(enabled=(device.type == 'cuda')):
                    outputs = model(features).squeeze(1)
                    val_loss += criterion(outputs, labels).item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        train_auc = roc_auc_score(train_labels, train_preds) if len(np.unique(train_labels)) > 1 else 0.5
        val_auc = roc_auc_score(val_labels, val_preds) if len(np.unique(val_labels)) > 1 else 0.5
        val_acc = accuracy_score(val_labels, (np.array(val_preds) > 0.5).astype(int))
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Train Loss: {avg_train_loss:.4f}, Train AUC: {train_auc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")

        log_df.loc[epoch] = [epoch+1, avg_train_loss, train_auc, avg_val_loss, val_auc, val_acc, scheduler.get_last_lr()[0]]
        log_df.to_csv(log_file, index=False)
        
        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print(f"\nTraining complete! Best validation AUC: {early_stopping.best_score:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training logs saved to: {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Audio Forensics Model")
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--epochs', type=int, default=CONFIG['num_epochs'])
    parser.add_argument('--lr', type=float, default=CONFIG['learning_rate'])
    args = parser.parse_args()

    CONFIG['batch_size'] = args.batch_size
    CONFIG['num_epochs'] = args.epochs
    CONFIG['learning_rate'] = args.lr

    main(args)