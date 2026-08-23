import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms ### Import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau ### Import scheduler
import librosa
from tqdm import tqdm
import warnings
import cv2


# --- Configuration ---
DATA_DIR = 'polyglot_processed_all_unbalanced'
IMAGE_SIZE = 224
MAX_FACES = 20
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
AUDIO_TARGET_LENGTH = 313
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 10

warnings.filterwarnings('ignore', 'PySoundFile failed. Trying audioread instead.')


class CrossAttention(nn.Module):
    """ Implements scaled dot-product cross-attention. """
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, query, key, value):
        q = self.query_proj(query).unsqueeze(1)
        k = self.key_proj(key)
        v = self.value_proj(value)

        attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        context = torch.bmm(attn_weights, v).squeeze(1)
        
        return context, attn_weights.squeeze(1)

class Attention(nn.Module):
    """
    A simple yet effective additive attention mechanism (Bahdanau-style).
    """
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.V = nn.Linear(input_dim, 1)

    def forward(self, lstm_outputs):
        """
        Forward pass for the attention layer.
        Args:
            lstm_outputs: The sequence of hidden states from the LSTM.
                          Shape: (batch_size, seq_len, lstm_hidden_dim)
        Returns:
            context_vector: The weighted sum of LSTM outputs.
                            Shape: (batch_size, lstm_hidden_dim)
            attention_weights: The calculated importance weights.
                               Shape: (batch_size, seq_len)
        """
        score = torch.tanh(self.W(lstm_outputs))
        attention_scores = self.V(score).squeeze(2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_outputs).squeeze(1)
        return context_vector, attention_weights

class CrossModal_CNN_LSTM(nn.Module):
    def __init__(self):
        super(CrossModal_CNN_LSTM, self).__init__()
        lstm_hidden_dim = 128
        
        # 1. Visual CNN-LSTM Pathway
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.cnn_base = nn.Sequential(*list(mobilenet.children())[:-1])
        for param in self.cnn_base.parameters(): param.requires_grad = False
        self.visual_lstm = nn.LSTM(input_size=1280, hidden_size=lstm_hidden_dim, 
                                   batch_first=True, bidirectional=True)

        # 2. Audio CNN-LSTM Pathway
        self.audio_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.audio_lstm = nn.LSTM(input_size=64 * 32, hidden_size=lstm_hidden_dim, 
                                  batch_first=True, bidirectional=True)

        # 3. Cross-Modal Attention Layer
        # The query will come from audio, keys/values from video.
        self.cross_attention = CrossAttention(embed_dim=lstm_hidden_dim * 2)

        # 4. Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, face_input, audio_input):
        # --- Process Visual Stream (to get Keys and Values) ---
        b, t, c, h, w = face_input.shape
        cnn_features = self.cnn_base(face_input.view(b * t, c, h, w)).mean([2, 3])
        visual_sequence, _ = self.visual_lstm(cnn_features.view(b, t, -1))

        # --- Process Audio Stream (to get the Query) ---
        audio_cnn_features = self.audio_cnn(audio_input).permute(0, 3, 1, 2)
        b_audio, w_audio, c_audio, h_audio = audio_cnn_features.shape
        # We need the final hidden state as our query
        _, (audio_h, _) = self.audio_lstm(audio_cnn_features.reshape(b_audio, w_audio, -1))
        # Concatenate final forward and backward hidden states
        audio_query = torch.cat((audio_h[-2,:,:], audio_h[-1,:,:]), dim=1)

        # --- Perform Cross-Attention ---
        # Use the audio summary to query the visual sequence
        cross_modal_context, cross_attention_weights = self.cross_attention(
            query=audio_query, 
            key=visual_sequence, 
            value=visual_sequence
        )
        
        # --- Classify ---
        output = self.classifier(cross_modal_context)
        
        return output, cross_attention_weights

class LipSyncDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and preprocessing lip-sync data.
    Applies a series of torchvision transforms to the visual data.
    """
    def __init__(self, directory, transforms=None):
        self.directory = directory
        self.file_list = [os.path.join(self.directory, f) for f in os.listdir(self.directory) if f.endswith('.npz')]
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        try:
            data = np.load(file_path, allow_pickle=True)

            # --- Process Visual Data ---
            loaded_faces = data['faces']
            num_faces_to_process = min(len(loaded_faces), MAX_FACES)
            padded_faces = torch.zeros((MAX_FACES, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)

            for i in range(num_faces_to_process):
                face = loaded_faces[i]
                if self.transforms:
                    transformed_face = self.transforms(face)
                    padded_faces[i] = transformed_face

            # --- Process Audio Data ---
            waveform = data['waveform'].astype('float32')
            log_mel_spec_padded = np.zeros((N_MELS, AUDIO_TARGET_LENGTH), dtype=np.float32)
            if waveform.size > 0:
                mel_spec = librosa.feature.melspectrogram(
                    y=waveform, sr=SAMPLE_RATE, n_fft=N_FFT,
                    hop_length=HOP_LENGTH, n_mels=N_MELS
                )
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

                if log_mel_spec.shape[1] > AUDIO_TARGET_LENGTH:
                    log_mel_spec_padded = log_mel_spec[:, :AUDIO_TARGET_LENGTH]
                else:
                    log_mel_spec_padded[:, :log_mel_spec.shape[1]] = log_mel_spec

            log_mel_spec_padded = torch.from_numpy(log_mel_spec_padded).unsqueeze(0)
            label = torch.tensor(1 if data['label'][0] == 'fake' else 0, dtype=torch.long)
            return padded_faces, log_mel_spec_padded, label

        except Exception as e:
            print(f"\nError loading or processing file {file_path}: {e}")
            return torch.zeros((MAX_FACES, 3, IMAGE_SIZE, IMAGE_SIZE)), \
                   torch.zeros((1, N_MELS, AUDIO_TARGET_LENGTH)), \
                   torch.tensor(0, dtype=torch.long)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ### 1. Define Data Augmentation & Normalization ###
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = LipSyncDataset(os.path.join(DATA_DIR, 'train'), transforms=train_transforms)
    val_dataset = LipSyncDataset(os.path.join(DATA_DIR, 'val'), transforms=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = CrossModal_CNN_LSTM().to(device)
    
    ### 2. Unfreeze layers for Fine-Tuning ###
    # Unfreeze the last 4 blocks of MobileNetV2's feature extractor
    for param in model.cnn_base[0][15:].parameters():
        param.requires_grad = True

    ### 3. Setup Optimizer with Differential LR ###
    # Create parameter groups for different learning rates
    finetune_params = list(model.cnn_base[0][15:].parameters())
    finetune_ids = {id(p) for p in finetune_params}
    base_params = [p for p in model.parameters() if id(p) not in finetune_ids and p.requires_grad]

    optimizer = optim.Adam([
        {'params': base_params},
        {'params': finetune_params, 'lr': LEARNING_RATE / 10} # Lower LR for fine-tuning
    ], lr=LEARNING_RATE)
    
    ### 4. Define LR Scheduler ###
    # torch >= 2.2 removed ReduceLROnPlateau's `verbose` argument
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3)

    criterion = nn.CrossEntropyLoss()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")

    best_val_accuracy = 0.0
    epochs_no_improve = 0
    
    print("\n--- Starting Model Training ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct_predictions, total_samples = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for faces, audio, labels in train_pbar:
            faces, audio, labels = faces.to(device), audio.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(faces, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * faces.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct_predictions/total_samples:.4f}'})

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / total_samples

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for faces, audio, labels in val_pbar:
                faces, audio, labels = faces.to(device), audio.to(device), labels.to(device)
                outputs, _ = model(faces, audio)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * faces.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}', 'val_acc': f'{val_correct/val_total:.4f}'})

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        print(f"Epoch {epoch+1} Summary: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

        if val_epoch_acc > best_val_accuracy:
            print(f"Validation accuracy improved from {best_val_accuracy:.4f} to {val_epoch_acc:.4f}. Saving model...")
            best_val_accuracy = val_epoch_acc
            torch.save(model.state_dict(), 'lip_sync_model_crossattention_all_unbalanced.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).")

        ### Scheduler Step ###
        scheduler.step(val_epoch_loss)

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered after {PATIENCE} epochs with no improvement.")
            break
            
    print("\n--- Training Complete ---")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")