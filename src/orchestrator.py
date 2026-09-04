
"""
Enhanced multi-agent system for deepfake detection with 5 agents:
Visual (Xception), Audio (Mel+CNN), Audio Forensics (ECAPA-TDNN), 
Cross-Modal (Lip-Sync), and Facial Biometric (Face Quality) analysis.
"""

import os
import logging
import warnings
from typing import Dict, Any, List, Optional
import cv2
import csv
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import librosa
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from speechbrain.pretrained import EncoderClassifier
from scipy import signal


from agents.visual_xception import XceptionDeepfakeDetector
from agents.cross_modal_lipsync import CrossModal_CNN_LSTM
from agents.audio_freqnet import FreqNet
from agents.audio_forensics_ecapa import OptimizedLightweightForensics, FastAudioFeatureExtractor


from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain.tools import StructuredTool


CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataloader_num_workers": 4,
    "data_dir": "data/polyglot_processed_all_unbalanced",
    "model_files": {
        "spatial": "checkpoints/xception/polyglotfake_xception_best_unbal_all_faceaug.pth",
        "audio": "checkpoints/freqnet/freqnet_model_all_unbalanced_improved.pth",
        "audio_forensics": "checkpoints/ecapa_forensic_head/audio_forensics_model_finetuned_best.pth",  
        "cross_modal": "checkpoints/cross_modal/lip_sync_model_crossattention.pth",
        "face_quality": "checkpoints/biometric/fine_tuning/best_model.pth",  # Add face quality model path
    },
    "audio_forensics_stats_path": "checkpoints/ecapa_forensic_head/training_stats.npz",
    "output_file": "analysis_results_with_5_agents.csv",
    "visual_agent": {
        "image_size": 299,
        "dropout_rate": 0.3,
    },
    "audio_agent": {
        "n_mels": 224,
        "sample_rate": 16000,
        "clip_duration_s": 5,
    },
    "audio_forensics_agent": {
        "sample_rate": 16000,
        "duration": 6,
        "embedding_dim": 192,
        "num_forensic_features": 11,
        "window_size": 2.0,
        "hop_size": 1.0,
    },
    "cross_modal_agent": {
        "max_faces": 20,
        "image_size": 224,
        "audio_target_length": 313,
        "sample_rate": 16000,
        "n_fft": 2048,
        "hop_length": 512,
        "n_mels": 128,
    },
    "face_quality_agent": {  
        "image_size": 299,
        "normalize": True,  
    },
    "decision_engine": {
        "weights": {
            "Visual (Spatial)": 0.20,
            "Audio (Mel+CNN)": 0.15,
            "Audio Forensics (ECAPA)": 0.20,
            "Cross-Modal (Lip-Sync)": 0.25,
            "Facial Biometric (Quality)": 0.20,  
        },
        "threshold": 0.37,
    }
}

# biometric
class FaceQualityNet(nn.Module):
    """Face quality assessment network - must match training architecture"""
    def __init__(self, num_classes: int = 1):
        super(FaceQualityNet, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=False)
        
        # 5 channel cnn 
        orig_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            5, orig_conv.out_channels, 
            kernel_size=orig_conv.kernel_size, 
            stride=orig_conv.stride, 
            padding=orig_conv.padding, 
            bias=False
        )
        
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.quality_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        quality_features = self.quality_head(features)
        output = self.classifier(quality_features)
        return torch.sigmoid(output)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)-8s] --- %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    warnings.filterwarnings("ignore")
    logging.getLogger("speechbrain").setLevel(logging.WARNING)

def check_system_readiness(split: str = "test") -> bool:
    data_split_dir = os.path.join(CONFIG["data_dir"], split)
    if not os.path.isdir(data_split_dir):
        logging.error(f"Data directory '{data_split_dir}' not found.")
        return False
    
    model_paths = list(CONFIG["model_files"].values())
    model_paths.append(CONFIG["audio_forensics_stats_path"])
    missing_files = [path for path in model_paths if not os.path.exists(path)]
    if missing_files:
        logging.error("One or more required files were not found:")
        for f in missing_files:
            logging.error(f"  - Missing: {f}")
        return False
    
    logging.info("System readiness check passed. All models and data directories found.")
    return True


def load_all_models() -> Dict[str, Any]:
    models = {}
    device = CONFIG['device']
    logging.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # xceptionnet model
    try:
        visual_model = XceptionDeepfakeDetector(
            num_classes=1, 
            dropout_rate=CONFIG["visual_agent"]["dropout_rate"], 
            pretrained=False
        ).to(device)
        checkpoint = torch.load(CONFIG["model_files"]["spatial"], map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        visual_model.load_state_dict(state_dict)
        visual_model.eval()
        models['spatial'] = visual_model
        logging.info("Visual (Xception) model loaded successfully.")
    except Exception as e:
        logging.error(f"Visual model failed: {e}")
        models['spatial'] = None
    
    # freqnet
    try:
        audio_model = FreqNet(num_classes=1).to(device)
        audio_model.load_state_dict(torch.load(CONFIG["model_files"]["audio"], map_location=device))
        audio_model.eval()
        models['audio'] = audio_model
        logging.info("Audio (Mel+CNN) model loaded successfully.")
    except Exception as e:
        logging.error(f"Audio (Mel+CNN) model failed: {e}")
        models['audio'] = None
    
    # Audio Forensics Model
    try:
        cfg_af = CONFIG['audio_forensics_agent']
        audio_forensics_model = OptimizedLightweightForensics(
            embedding_dim=cfg_af['embedding_dim'], 
            num_forensic_features=cfg_af['num_forensic_features']
        ).to(device)
        audio_forensics_model.load_state_dict(
            torch.load(CONFIG["model_files"]["audio_forensics"], map_location=device)
        )
        audio_forensics_model.eval()
        models['audio_forensics'] = audio_forensics_model
        
        models['speaker_encoder'] = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="checkpoints/speechbrain_cache", 
            run_opts={"device": str(device)}
        )
        models['speaker_encoder'].eval()
        
        stats = np.load(CONFIG['audio_forensics_stats_path'])
        models['audio_forensics_stats'] = {
            'mean': torch.tensor(stats['mean'], dtype=torch.float32, device=device),
            'std': torch.tensor(stats['std'], dtype=torch.float32, device=device)
        }
        
        logging.info("Audio Forensics (ECAPA-TDNN) model and stats loaded successfully.")
    except Exception as e:
        logging.error(f"Audio Forensics model failed: {e}", exc_info=True)
        models['audio_forensics'] = None
        models['speaker_encoder'] = None

    # Cross-Modal Model
    try:
        lip_sync_model = CrossModal_CNN_LSTM().to(device)
        lip_sync_model.load_state_dict(
            torch.load(CONFIG["model_files"]["cross_modal"], map_location=device)
        )
        lip_sync_model.eval()
        models['cross_modal'] = lip_sync_model
        logging.info("Cross-Modal (Lip-Sync) model loaded successfully.")
    except Exception as e:
        logging.error(f"Cross-modal model failed: {e}")
        models['cross_modal'] = None
    
    # Face Quality Model
    try:
        face_quality_model = FaceQualityNet(num_classes=1).to(device)
        checkpoint = torch.load(CONFIG["model_files"]["face_quality"], map_location=device)
        face_quality_model.load_state_dict(checkpoint['model_state_dict'])
        face_quality_model.eval()
        models['face_quality'] = face_quality_model
        logging.info("Facial Biometric (Face Quality) model loaded successfully.")
    except Exception as e:
        logging.error(f"Face Quality model failed: {e}", exc_info=True)
        models['face_quality'] = None
    
    return models


def create_error_result(agent_name: str, details: str) -> Dict[str, Any]:
    return {'agent': agent_name, 'score': -1.0, 'details': details, 'error': True}


def run_visual_analysis(media_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """Analyzes a single frame for spatial artifacts using PyTorch Xception model."""
    agent_name = "Visual (Spatial)"
    model = models.get('spatial')
    device = CONFIG['device']
    
    if model is None: 
        return create_error_result(agent_name, "Model not loaded.")
    
    faces = media_data.get('processed_faces')
    if faces is None or len(faces) == 0:
        return create_error_result(agent_name, "No faces found.")
    
    try:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((CONFIG["visual_agent"]["image_size"], CONFIG["visual_agent"]["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        face = faces[0]
        if torch.is_tensor(face):
            face = face.cpu().numpy()
        
        if face.dtype != np.uint8:
            face = (face * 255).astype(np.uint8)
        
        input_tensor = transform(face).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.sigmoid(logits)
            score = probabilities.squeeze().item()
        
        return {
            'agent': agent_name, 
            'score': float(score), 
            'details': 'PyTorch Xception analysis.'
        }
        
    except Exception as e:
        logging.error(f"Visual analysis failed: {e}", exc_info=True)
        return create_error_result(agent_name, f"Analysis error: {e}")

def run_audio_analysis(media_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """Analyzes an audio waveform using the PyTorch FreqNet model."""
    agent_name = "Audio (Mel+CNN)"
    model = models.get('audio')
    cfg, device = CONFIG["audio_agent"], CONFIG["device"]
    
    if model is None: 
        return create_error_result(agent_name, "Model not loaded.")
    
    waveform = media_data.get('audio_waveform')
    if waveform is None or (torch.is_tensor(waveform) and waveform.numel() < 400) or (isinstance(waveform, np.ndarray) and waveform.size < 400):
        return create_error_result(agent_name, "Insufficient audio data.")

    try:
        waveform = waveform.cpu().numpy() if torch.is_tensor(waveform) else waveform
        sr = cfg['sample_rate']
        target_length = sr * cfg['clip_duration_s']
        
        if len(waveform) > target_length: 
            y = waveform[:target_length]
        else: 
            y = np.pad(waveform, (0, target_length - len(waveform)), 'constant')
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=cfg["n_mels"])
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        if log_mel_spec.max() > log_mel_spec.min():
            norm_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
        else:
            norm_spec = np.zeros_like(log_mel_spec)
        
        input_tensor = torch.tensor(norm_spec, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        score = torch.sigmoid(output).item()
        return {'agent': agent_name, 'score': score, 'details': 'Mel+CNN model analysis.'}
        
    except Exception as e:
        logging.error(f"AudioAgent analysis failed: {e}", exc_info=True)
        return create_error_result(agent_name, f"Analysis error: {e}")

def run_cross_modal_analysis(media_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """Analyzes audio-lip sync coherence using a PyTorch lip-sync model."""
    agent_name = "Cross-Modal (Lip-Sync)"
    model = models.get('cross_modal')
    cfg, device = CONFIG["cross_modal_agent"], CONFIG["device"]
    
    if model is None: 
        return create_error_result(agent_name, "Model not loaded.")
    
    faces = media_data.get('processed_faces')
    waveform = media_data.get('audio_waveform')
    
    faces_empty = (faces is None or 
                   (hasattr(faces, 'size') and faces.size == 0) or 
                   (hasattr(faces, 'shape') and len(faces) == 0))
    
    waveform_insufficient = (waveform is None or 
                             (torch.is_tensor(waveform) and waveform.numel() < 4096) or
                             (isinstance(waveform, np.ndarray) and waveform.size < 4096))
    
    if faces_empty or waveform_insufficient:
        return create_error_result(agent_name, "Insufficient visual or audio data.")

    try:
        if torch.is_tensor(waveform):
            waveform = waveform.cpu().numpy()
            
        visual_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((cfg["image_size"], cfg["image_size"]), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        num_faces = min(len(faces), cfg["max_faces"])
        padded_faces = torch.zeros((cfg["max_faces"], 3, cfg["image_size"], cfg["image_size"]), dtype=torch.float32)
        
        for i in range(num_faces):
            face = faces[i]
            if torch.is_tensor(face):
                face = face.cpu().numpy()
            
            if face.dtype != np.uint8:
                face = (face * 255).astype(np.uint8) if face.max() <= 1.0 else face.astype(np.uint8)
            
            padded_faces[i] = visual_transforms(face)
        
        visual_tensor = padded_faces.unsqueeze(0).to(device)

        mel_spec = librosa.feature.melspectrogram(
            y=waveform, sr=cfg["sample_rate"], n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"], n_mels=cfg["n_mels"]
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec_padded = np.zeros((cfg["n_mels"], cfg["audio_target_length"]), dtype=np.float32)
        
        if log_mel_spec.shape[1] > cfg["audio_target_length"]:
            log_mel_spec_padded = log_mel_spec[:, :cfg["audio_target_length"]]
        else:
            log_mel_spec_padded[:, :log_mel_spec.shape[1]] = log_mel_spec
        
        audio_tensor = torch.from_numpy(log_mel_spec_padded).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output, _ = model(visual_tensor, audio_tensor)
            probabilities = torch.softmax(output, dim=1)
            score = probabilities[0, 1].item()

        return {'agent': agent_name, 'score': score, 'details': 'PyTorch lip-sync coherence analysis.'}
        
    except Exception as e:
        logging.error(f"CrossModalAgent analysis failed: {e}", exc_info=True)
        return create_error_result(agent_name, f"Analysis error: {e}")

def run_audio_forensics_analysis(media_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """Analyzes audio using ECAPA-TDNN embeddings with correct normalization."""
    agent_name = "Audio Forensics (ECAPA)"
    model = models.get('audio_forensics')
    speaker_encoder = models.get('speaker_encoder')
    stats = models.get('audio_forensics_stats')
    cfg, device = CONFIG["audio_forensics_agent"], CONFIG['device']
    
    if model is None or speaker_encoder is None or stats is None: 
        return create_error_result(agent_name, "Model, encoder or stats not loaded.")
    
    waveform = media_data.get('audio_waveform')
    if waveform is None or (torch.is_tensor(waveform) and waveform.numel() < 400) or (isinstance(waveform, np.ndarray) and waveform.size < 400):
        return create_error_result(agent_name, "Insufficient audio data.")
    
    try:
        if torch.is_tensor(waveform):
            waveform = waveform.cpu().numpy()
        
        sr = cfg['sample_rate']
        feature_extractor = FastAudioFeatureExtractor()
        target_length = sr * cfg['duration']
        
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)), 'constant')
        
        prosody_features = feature_extractor.extract_fast_prosody(waveform, sr)
        artifact_features = feature_extractor.extract_fast_artifacts(waveform, sr)
        
        with torch.no_grad(), autocast():
            waveform_tensor_full = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
            embedding = speaker_encoder.encode_batch(waveform_tensor_full).squeeze()

            window_size = int(cfg['window_size'] * sr)
            embeddings_list = []
            for pos in [0, len(waveform)//2 - window_size//2, len(waveform) - window_size]:
                if pos >= 0 and pos + window_size <= len(waveform):
                    window_tensor = torch.tensor(waveform[pos:pos + window_size], dtype=torch.float32).unsqueeze(0).to(device)
                    embeddings_list.append(speaker_encoder.encode_batch(window_tensor).squeeze())

            if len(embeddings_list) >= 2:
                embeddings_stack = torch.stack(embeddings_list)
                distances = torch.norm(embeddings_stack[:-1] - embeddings_stack[1:], dim=1)
                temporal_features = torch.tensor([torch.mean(distances), torch.std(distances)], device=device)
                embedding_var = torch.std(embeddings_stack, dim=0).mean()
            else:
                temporal_features = torch.zeros(2, device=device)
                embedding_var = torch.tensor(0.0, device=device)

        all_features = torch.cat([
            embedding, 
            torch.tensor(prosody_features, device=device), 
            torch.tensor(artifact_features, device=device), 
            temporal_features, 
            embedding_var.unsqueeze(0)
        ])
        
        normalized_features = (all_features - stats['mean']) / (stats['std'] + 1e-6)
        
        input_tensor = normalized_features.unsqueeze(0)
        with torch.no_grad(), autocast():
            output = model(input_tensor)
            score = torch.sigmoid(output).squeeze().item()
        
        return {'agent': agent_name, 'score': float(score), 'details': 'ECAPA-TDNN voice forensics analysis.'}
    except Exception as e:
        logging.error(f"Audio Forensics analysis failed: {e}", exc_info=True)
        return create_error_result(agent_name, f"Analysis error: {e}")


def run_face_quality_analysis(media_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """Analyzes face quality metrics for deepfake detection."""
    agent_name = "Facial Biometric (Quality)"
    model = models.get('face_quality')
    cfg, device = CONFIG["face_quality_agent"], CONFIG['device']
    
    if model is None:
        return create_error_result(agent_name, "Model not loaded.")
    
    faces = media_data.get('processed_faces')
    if faces is None or len(faces) == 0:
        return create_error_result(agent_name, "No faces found.")
    
    try:
        image_size = cfg['image_size']
        
        # Select middle face for quality assessment
        face_idx = len(faces) // 2 if len(faces) > 1 else 0
        face = faces[face_idx]
        
        # Convert to numpy if tensor
        if torch.is_tensor(face):
            face = face.cpu().numpy()
        
        # Ensure uint8 format
        if face.dtype != np.uint8:
            face = (face * 255).astype(np.uint8) if face.max() <= 1.0 else face.astype(np.uint8)
        
        # Extract quality metrics
        def extract_quality_metrics(face_img):
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
            
            # Blur score (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_tensor = torch.full((1, image_size, image_size), blur_score / 1000.0)
            
            # Exposure score (mean brightness)
            exposure_score = np.mean(gray) / 255.0
            exposure_tensor = torch.full((1, image_size, image_size), exposure_score)
            
            return torch.cat([blur_tensor, exposure_tensor], dim=0)
        
        # Prepare transforms
        base_transforms = [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        
        if cfg['normalize']:
            base_transforms.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        
        transform = transforms.Compose(base_transforms)
        
        # Process face
        face_tensor = transform(face)
        quality_features = extract_quality_metrics(face)
        
        # Combine face tensor and quality features (5 channels total)
        combined_input = torch.cat([face_tensor, quality_features], dim=0)
        
        # Add batch dimension and move to device
        input_tensor = combined_input.unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            score = output.squeeze().item()
        
        return {
            'agent': agent_name,
            'score': float(score),
            'details': 'Face quality biometric analysis.'
        }
        
    except Exception as e:
        logging.error(f"Face Quality analysis failed: {e}", exc_info=True)
        return create_error_result(agent_name, f"Analysis error: {e}")

class DeepfakeNPZDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        try:
            data = np.load(filepath, allow_pickle=True)
            faces = data.get('faces', np.array([]))
            waveform = data.get('waveform', np.array([]))
            label = data.get('label', ["Unknown"])[0]
            return {'faces': faces, 'waveform': waveform, 'label': label, 'filepath': filepath}
        except Exception as e:
            logging.error(f"Failed to load .npz file '{filepath}': {e}")
            return {'faces': np.array([]), 'waveform': np.array([]), 'label': "Error", 'filepath': filepath}

def process_batch(batch: dict, models: dict) -> Optional[Dict[str, Any]]:
    filepath = batch['filepath'][0]
    label = batch['label'][0]
    
    if label == "Error":
        return None
        
    
    media_data = {
        'filepath': filepath,
        'processed_faces': batch['faces'].squeeze(0),
        'audio_waveform': batch['waveform'].squeeze(0),
        'audio_sample_rate': CONFIG["audio_agent"]["sample_rate"],
        'metadata': {'ground_truth': label.capitalize()}
    }
    
    lc_input = {"media_data": media_data, "models": models}
    return lc_input

# aggregation for 5 agents
def aggregate_results(analysis_outputs: Dict[str, Any]) -> Dict[str, Any]:
    results = [
        analysis_outputs.get('visual'), 
        analysis_outputs.get('audio'), 
        analysis_outputs.get('audio_forensics'), 
        analysis_outputs.get('cross_modal'),
        analysis_outputs.get('face_quality')  # Add face quality results
    ]
    results = [r for r in results if r is not None]
    decision_cfg = CONFIG["decision_engine"]
    valid_results = [r for r in results if not r.get('error')]
    
    if not valid_results:
        final_decision = {"verdict": "Uncertain", "confidence": 0.0, "reason": "All agents failed.", "results": results}
    else:
        weighted_sum, total_weight = 0, 0
        for r in valid_results:
            weight = decision_cfg["weights"].get(r['agent'], 0)
            if weight > 0:
                weighted_sum += r['score'] * weight
                total_weight += weight
        final_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0
        verdict = "Deepfake" if final_score >= decision_cfg["threshold"] else "Real"
        reason = f"Weighted score ({final_score:.3f}) is {'above' if verdict == 'Deepfake' else 'below'} threshold ({decision_cfg['threshold']})."
        final_decision = {"verdict": verdict, "confidence": final_score, "reason": reason, "results": results}
    final_decision['media_data'] = analysis_outputs['original_data']['media_data']
    return final_decision

def print_report(decision: Dict[str, Any]):
    media_data = decision['media_data']
    report = f"""
----------------------------------------------------------------------
  Analysis Report for: {os.path.basename(media_data['filepath'])}
  Ground Truth: {media_data['metadata']['ground_truth']}
  System Verdict: {decision['verdict']} (Confidence: {decision['confidence']:.2%})
  Reason: {decision['reason']}
----------------------------------------------------------------------
  Agent Breakdown (5 Agents):"""
    for result in decision.get('results', []):
        if result:
            score_str = f'{result["score"]:.3f}' if not result.get('error') else 'N/A'
            report += f"\n    - {result['agent']:<30} | Score: {score_str:<7} | Status: {result['details']}"
    report += "\n----------------------------------------------------------------------"
    logging.info(report)

class Evaluator:
    def __init__(self):
        self.y_true, self.y_pred = [], []
        self.label_map = {'Real': 0, 'Fake': 1}
        self.verdict_map = {'Real': 0, 'Deepfake': 1}
        logging.info("Evaluator initialized.")
    
    def log_result(self, ground_truth: str, system_verdict: str):
        true_val, pred_val = self.label_map.get(ground_truth), self.verdict_map.get(system_verdict)
        if true_val is not None and pred_val is not None:
            self.y_true.append(true_val)
            self.y_pred.append(pred_val)
    
    def display_report(self):
        if not self.y_true:
            logging.warning("No results were logged.")
            return
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision = precision_score(self.y_true, self.y_pred, zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, zero_division=0)
        cm = confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])
        print(f"""
======================================================================
                    ENHANCED EVALUATION REPORT (5 Agents)
======================================================================
  Total Samples Processed: {len(self.y_true)}
  - Accuracy:  {accuracy:.2%}
  - Precision: {precision:.2%}
  - Recall:    {recall:.2%}
  - F1-Score:  {f1:.2%}
  - Confusion Matrix:
                 Predicted
                -----------------
               |  Real |  Fake |
    -------------------------
    Actual Real | {cm[0][0]:^5} | {cm[0][1]:^5} |
    Actual Fake | {cm[1][0]:^5} | {cm[1][1]:^5} |
    -------------------------
======================================================================""")

def main():
    parser = argparse.ArgumentParser(
        description="Score a preprocessed split (train/val/test) with the 5-agent "
                    "weighted orchestrator and write the per-sample CSV used by the "
                    "paper artifacts. Use --split val to generate the validation scores "
                    "that justify the operating-point selection (see "
                    "paper_artifacts/task_00_select_operating_point.py)."
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Which split under CONFIG['data_dir'] to score (default: test).")
    parser.add_argument("--output_file", default=None,
                        help="Output CSV path (default: CONFIG['output_file']).")
    args = parser.parse_args()
    output_file = args.output_file or CONFIG["output_file"]

    setup_logging()

    print("\n" + "="*60)
    print("Enhanced Multi-Agent Deepfake Detection System")
    print("Agents: Visual, Audio (Mel+CNN), Audio Forensics,")
    print("        Cross-Modal, and Facial Biometric Quality")
    print("="*60 + "\n")
    
    if not check_system_readiness(args.split):
        logging.critical("System readiness check failed. Exiting.")
        return
    
    models = load_all_models()
    

    analysis_branch = RunnableParallel(
        visual=StructuredTool.from_function(run_visual_analysis),
        audio=StructuredTool.from_function(run_audio_analysis),
        audio_forensics=StructuredTool.from_function(run_audio_forensics_analysis),
        cross_modal=StructuredTool.from_function(run_cross_modal_analysis),
        face_quality=StructuredTool.from_function(run_face_quality_analysis),  # Add new agent
        original_data=RunnablePassthrough()
    )
    
    full_analysis_chain = (
        analysis_branch |
        RunnableLambda(aggregate_results)
    )
    
    test_dir = os.path.join(CONFIG["data_dir"], args.split)
    test_files = [os.path.join(root, file) for root, _, files in os.walk(test_dir)
                  for file in files if file.lower().endswith('.npz')]

    if not test_files:
        logging.warning(f"No .npz files found in '{test_dir}'. Cannot run evaluation.")
        return
        
    dataset = DeepfakeNPZDataset(test_files)
    data_loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=CONFIG['dataloader_num_workers'], 
        pin_memory=True
    )
    
    evaluator = Evaluator()
    all_results = []
    

    total_files = len(test_files)
    processed = 0
    
    print(f"\nProcessing {total_files} {args.split} files...")
    print("-" * 60)

    for batch in data_loader:
        try:

            lc_input = process_batch(batch, models)
            
            if lc_input:
                final_decision = full_analysis_chain.invoke(lc_input)
                
            if final_decision:
                print_report(final_decision)
                evaluator.log_result(
                    ground_truth=final_decision['media_data']['metadata']['ground_truth'],
                    system_verdict=final_decision['verdict']
                )

                result_row = {
                    'filepath': os.path.basename(final_decision['media_data']['filepath']),
                    'ground_truth': final_decision['media_data']['metadata']['ground_truth'],
                    'system_verdict': final_decision['verdict'],
                    'final_score': final_decision['confidence']
                }
                
                
                for agent_result in final_decision.get('results', []):
                    if agent_result:
                        agent_name = agent_result.get('agent', 'Unknown Agent')
                        score = agent_result.get('score', -1.0)
                        result_row[f'score_{agent_name}'] = score
                
                all_results.append(result_row)
            
            
            processed += 1
            if processed % 10 == 0:
                print(f"Progress: {processed}/{total_files} files processed ({processed/total_files*100:.1f}%)")
                
        except Exception as e:
            logging.error(f"Error processing batch: {e}", exc_info=True)
            continue

    print("\n" + "="*60)
    evaluator.display_report()
    print("="*60)
    
    
    if not all_results:
        logging.warning("No results were collected to save to CSV.")
        return

    try:
        fieldnames = all_results[0].keys()
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
            
        logging.info(f"\nAnalysis results for {len(all_results)} files saved to {output_file}")

    except Exception as e:
        logging.error(f"Failed to save results to CSV: {e}")

if __name__ == "__main__":
    main()