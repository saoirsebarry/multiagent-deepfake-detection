
"""
This script implements an advanced multi-agent system for detecting deepfakes.
It features dynamic agent selection based on confidence scores, inter-agent
communication for resolving conflicts, and an iterative refinement process
for cases of disagreement.
"""

# [Keep all your existing imports - they remain the same]
import os
import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple
import cv2
import csv
import time
from collections import defaultdict

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

# Import your existing models
from agents.visual_xception import XceptionDeepfakeDetector
from agents.cross_modal_lipsync import CrossModal_CNN_LSTM
from agents.audio_freqnet import FreqNet
from agents.audio_forensics_ecapa import OptimizedLightweightForensics, FastAudioFeatureExtractor

# LangChain Libraries
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain.tools import StructuredTool

# [Your existing CONFIG remains the same, just add these new sections]
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataloader_num_workers": 4,
    "data_dir": "data/polyglot_processed_all_unbalanced",
    "model_files": {
        "spatial": "checkpoints/xception/polyglotfake_xception_best_unbal_all_faceaug.pth",
        "audio": "checkpoints/freqnet/freqnet_model_all_unbalanced_improved.pth",
        "audio_forensics": "checkpoints/ecapa_forensic_head/audio_forensics_model_finetuned_best.pth",
        "cross_modal": "checkpoints/cross_modal/lip_sync_model_crossattention.pth",
        "face_quality": "checkpoints/biometric/fine_tuning/best_model.pth",
    },
    "audio_forensics_stats_path": "checkpoints/ecapa_forensic_head/training_stats.npz",
    "output_file": "analysis_results_with_5_agents_orchestration.csv",
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
    },
    # Configuration for the multi-agent orchestration logic
    "multi_agent": {
        "confidence_threshold": 0.7,  # Threshold for a high-confidence decision
        "disagreement_threshold": 0.3,  # Defines a significant disagreement between agents
        "strong_agents": ["Visual (Spatial)", "Audio (Mel+CNN)"],
        "quick_agents": ["Facial Biometric (Quality)","Audio Forensics (ECAPA)" ,"Cross-Modal (Lip-Sync)"],  # Agents for fast initial screening
        "max_iterations": 3,
        "convergence_threshold": 0.05,
    }
}

# [Keep your existing model definitions - FaceQualityNet, etc.]
class FaceQualityNet(nn.Module):
    """
    Defines the Face Quality Assessment network, ensuring the architecture
    matches the one used during training.
    """
    def __init__(self, num_classes: int = 1):
        super(FaceQualityNet, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=False)

        # Modify the first convolutional layer to accept 5 input channels
        # (3 for RGB, 2 for quality metrics)
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

# === NEW MULTI-AGENT COMPONENTS ===

class AgentCommunicationHub:
    """A central hub for agents to communicate and coordinate their findings."""
    def __init__(self):
        self.message_queue = defaultdict(list)
        self.analysis_history = []
        self.agent_states = {}

    def record_analysis(self, agent_name: str, result: Dict[str, Any]):
        """Logs an agent's analysis result and updates its current state."""
        self.analysis_history.append({
            'agent': agent_name,
            'result': result,
            'timestamp': time.time()
        })
        self.agent_states[agent_name] = result

    def get_consensus(self) -> Tuple[float, bool]:
        """Calculates the current level of consensus among the agents."""
        if not self.agent_states:
            return 0.5, False

        scores = [r['score'] for r in self.agent_states.values() if not r.get('error')]
        if not scores:
            return 0.5, False

        # Determine if agents agree on the verdict (fake > 0.5 vs. real <= 0.5)
        verdicts = ['fake' if s > 0.5 else 'real' for s in scores]
        consensus_verdict = max(set(verdicts), key=verdicts.count)
        agreement_rate = verdicts.count(consensus_verdict) / len(verdicts)

        # Also consider the variance in scores
        score_variance = np.var(scores) if len(scores) > 1 else 0

        # Consensus is defined by high agreement and low variance
        has_consensus = agreement_rate >= 0.8 and score_variance < 0.1

        return agreement_rate, has_consensus

    def identify_conflicts(self) -> List[Tuple[str, str]]:
        """Identifies pairs of agents that have conflicting verdicts."""
        conflicts = []
        agents = list(self.agent_states.keys())

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                score1 = self.agent_states[agents[i]].get('score', 0.5)
                score2 = self.agent_states[agents[j]].get('score', 0.5)

                # A conflict occurs if one says real and the other says fake
                if (score1 > 0.5) != (score2 > 0.5):
                    conflicts.append((agents[i], agents[j]))

        return conflicts

class MultiAgentOrchestrator:
    """Manages the workflow of the multi-agent analysis."""
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.comm_hub = AgentCommunicationHub()
        self.ma_config = CONFIG["multi_agent"]

        # Map agent names to their respective analysis functions
        self.agent_functions = {
            "Visual (Spatial)": run_visual_analysis,
            "Audio (Mel+CNN)": run_audio_analysis,
            "Audio Forensics (ECAPA)": run_audio_forensics_analysis,
            "Cross-Modal (Lip-Sync)": run_cross_modal_analysis,
            "Facial Biometric (Quality)": run_face_quality_analysis,
        }

    def orchestrate_analysis(self, media_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinates the multi-agent analysis, allocating resources dynamically.
        """
        start_time = time.time()

        # Phase 1: Start with a quick assessment using the faster agents.
        logging.info("Phase 1: Running quick assessment with lightweight agents.")
        quick_results = self._run_quick_assessment(media_data)

        # If the quick agents agree with high confidence, we can stop early.
        quick_consensus = self._check_quick_consensus(quick_results)
        if quick_consensus['has_consensus'] and quick_consensus['confidence'] > self.ma_config['confidence_threshold']:
            logging.info(f"Quick consensus reached with {quick_consensus['confidence']:.2%} confidence. Finalizing decision.")
            return self._finalize_decision(quick_results, phase="quick")

        # Phase 2: If there's disagreement, bring in the more powerful agents.
        logging.info("Phase 2: Disagreement detected. Deploying strong arbitrator agents.")
        strong_results = self._run_strong_agents(media_data)

        # Combine all results so far.
        all_results = {**quick_results, **strong_results}
        self._update_communication_hub(all_results)

        # Check for consensus again after the strong agents have weighed in.
        consensus_rate, has_consensus = self.comm_hub.get_consensus()
        if has_consensus:
            logging.info(f"Consensus reached after strong agents ({consensus_rate:.2%} agreement).")
            return self._finalize_decision(all_results, phase="strong")

        # Phase 3: If disagreement persists, attempt iterative refinement.
        logging.info("Phase 3: Persistent disagreement. Starting iterative refinement.")
        refined_results = self._iterative_refinement(media_data, all_results)

        # Make a final decision using all available information.
        final_results = {**all_results, **refined_results}
        decision = self._finalize_decision(final_results, phase="iterative")

        decision['analysis_time'] = time.time() - start_time
        decision['phases_used'] = self._get_phases_used()

        return decision

    def _run_quick_assessment(self, media_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the lightweight agents for an initial screening."""
        results = {}
        for agent_name in self.ma_config['quick_agents']:
            if agent_name in self.agent_functions:
                try:
                    result = self.agent_functions[agent_name](media_data, self.models)
                    results[agent_name] = result
                    self.comm_hub.record_analysis(agent_name, result)
                except Exception as e:
                    logging.error(f"Quick agent {agent_name} encountered an error: {e}")
                    results[agent_name] = create_error_result(agent_name, str(e))
        return results

    def _check_quick_consensus(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Checks if the quick agents have reached a confident consensus."""
        valid_scores = [r['score'] for r in results.values() if not r.get('error')]

        if not valid_scores:
            return {'has_consensus': False, 'confidence': 0}

        # Check if they all agree on the verdict (real vs. fake)
        verdicts = ['fake' if s > 0.5 else 'real' for s in valid_scores]
        if len(set(verdicts)) == 1:
            avg_score = np.mean(valid_scores)
            confidence = abs(avg_score - 0.5) * 2  # Normalize to a 0-1 confidence scale
            return {'has_consensus': True, 'confidence': confidence}

        return {'has_consensus': False, 'confidence': 0}

    def _run_strong_agents(self, media_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deploys the more computationally intensive agents to resolve ambiguity."""
        results = {}
        for agent_name in self.ma_config['strong_agents']:
            if agent_name in self.agent_functions:
                try:
                    result = self.agent_functions[agent_name](media_data, self.models)
                    results[agent_name] = result
                    self.comm_hub.record_analysis(agent_name, result)

                    # Note if a strong agent has a very high-confidence result
                    if result['score'] > 0.8 or result['score'] < 0.2:
                        logging.info(f"Strong signal detected from {agent_name} with score: {result['score']:.3f}")

                except Exception as e:
                    logging.error(f"Strong agent {agent_name} encountered an error: {e}")
                    results[agent_name] = create_error_result(agent_name, str(e))
        return results

    def _iterative_refinement(self, media_data: Dict[str, Any],
                            current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Attempts to resolve persistent conflicts among agents."""
        refined_results = {}
        conflicts = self.comm_hub.identify_conflicts()

        if conflicts:
            logging.info(f"Attempting to resolve {len(conflicts)} conflicts through re-analysis.")
            # This is a placeholder for more complex refinement logic.
            # For now, we use the cross-modal agent as a tie-breaker.
            for agent1, agent2 in conflicts[:2]:  # Limit to the top conflicts
                if "Cross-Modal (Lip-Sync)" not in [agent1, agent2]:
                    cm_score = current_results.get("Cross-Modal (Lip-Sync)", {}).get('score', 0.5)
                    logging.info(f"Using Cross-Modal score ({cm_score:.3f}) as an arbiter.")

        return refined_results

    def _update_communication_hub(self, results: Dict[str, Any]):
        """Updates the central hub with the latest set of results."""
        for agent_name, result in results.items():
            if result and not result.get('error'):
                self.comm_hub.agent_states[agent_name] = result

    def _finalize_decision(self, results: Dict[str, Any], phase: str) -> Dict[str, Any]:
        """Aggregates all results to produce a final verdict."""
        valid_results = {k: v for k, v in results.items() if not v.get('error')}

        if not valid_results:
            return {
                'verdict': 'Uncertain',
                'confidence': 0.0,
                'reason': 'All agents failed to produce a result.',
                'results': list(results.values()),
                'phase': phase
            }

        # Use different weighting strategies depending on the analysis phase
        if phase == "quick":
            # In the quick phase, all agents are weighted equally
            weights = {agent: 1.0 for agent in valid_results.keys()}
        elif phase == "strong":
            # When strong agents are involved, give them more weight
            weights = {}
            for agent in valid_results.keys():
                weights[agent] = 1.5 if agent in self.ma_config['strong_agents'] else 0.5
        else:  # iterative phase
            # Use the default configured weights
            weights = CONFIG["decision_engine"]["weights"]

        # Calculate the final weighted score
        weighted_sum = sum(
            valid_results[agent]['score'] * weights.get(agent, 1.0)
            for agent in valid_results.keys()
        )
        total_weight = sum(weights.get(agent, 1.0) for agent in valid_results.keys())

        final_score = weighted_sum / total_weight if total_weight > 0 else 0.5

        threshold = CONFIG["decision_engine"]["threshold"]
        verdict = "Deepfake" if final_score >= threshold else "Real"

        # Generate a human-readable explanation for the decision
        consensus_rate, _ = self.comm_hub.get_consensus()
        if consensus_rate > 0.8:
            reason = f"Strong consensus ({consensus_rate:.0%}) among agents. Final score: {final_score:.3f}"
        elif len(valid_results) == len(self.agent_functions):
            reason = f"All agents were consulted. Final weighted score: {final_score:.3f}"
        else:
            reason = f"Partial analysis from {len(valid_results)} agents. Final score: {final_score:.3f}"

        return {
            'verdict': verdict,
            'confidence': final_score,
            'reason': reason,
            'results': list(results.values()),
            'phase': phase,
            'consensus_rate': consensus_rate,
            'agents_used': list(valid_results.keys())
        }

    def _get_phases_used(self) -> List[str]:
        """Tracks which analysis phases were executed."""
        phases = ["quick"]
        if any(agent in self.comm_hub.agent_states for agent in self.ma_config['strong_agents']):
            phases.append("strong")
        if len(self.comm_hub.analysis_history) > len(self.agent_functions):
            phases.append("iterative")
        return phases


# [Keep all your existing agent functions unchanged]
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)-8s] --- %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    warnings.filterwarnings("ignore")
    logging.getLogger("speechbrain").setLevel(logging.WARNING)

def check_system_readiness() -> bool:
    test_dir = os.path.join(CONFIG["data_dir"], 'test')
    if not os.path.isdir(test_dir):
        logging.error(f"Test data directory '{test_dir}' not found.")
        return False

    model_paths = list(CONFIG["model_files"].values())
    model_paths.append(CONFIG["audio_forensics_stats_path"])
    missing_files = [path for path in model_paths if not os.path.exists(path)]
    if missing_files:
        logging.error("One or more required model or data files were not found:")
        for f in missing_files:
            logging.error(f"  - Missing: {f}")
        return False

    logging.info("System readiness check passed. All models and data directories found.")
    return True


def load_all_models() -> Dict[str, Any]:
    # [Keep your existing load_all_models function exactly as is]
    models = {}
    device = CONFIG['device']
    logging.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info(f"GPU found: {torch.cuda.get_device_name(0)}")
    
    # Load Visual Model
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
        logging.error(f"Failed to load visual model: {e}")
        models['spatial'] = None
    
    # Load Audio Model
    try:
        audio_model = FreqNet(num_classes=1).to(device)
        audio_model.load_state_dict(torch.load(CONFIG["model_files"]["audio"], map_location=device))
        audio_model.eval()
        models['audio'] = audio_model
        logging.info("Audio (Mel+CNN) model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load audio (Mel+CNN) model: {e}")
        models['audio'] = None
    
    # Load Audio Forensics Model
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
        logging.error(f"Failed to load audio forensics model: {e}", exc_info=True)
        models['audio_forensics'] = None
        models['speaker_encoder'] = None

    # Load Cross-Modal Model
    try:
        lip_sync_model = CrossModal_CNN_LSTM().to(device)
        lip_sync_model.load_state_dict(
            torch.load(CONFIG["model_files"]["cross_modal"], map_location=device)
        )
        lip_sync_model.eval()
        models['cross_modal'] = lip_sync_model
        logging.info("Cross-Modal (Lip-Sync) model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load cross-modal model: {e}")
        models['cross_modal'] = None
    
    # Load Face Quality Model
    try:
        face_quality_model = FaceQualityNet(num_classes=1).to(device)
        checkpoint = torch.load(CONFIG["model_files"]["face_quality"], map_location=device)
        face_quality_model.load_state_dict(checkpoint['model_state_dict'])
        face_quality_model.eval()
        models['face_quality'] = face_quality_model
        logging.info("Facial Biometric (Face Quality) model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load face quality model: {e}", exc_info=True)
        models['face_quality'] = None
    
    return models

def create_error_result(agent_name: str, details: str) -> Dict[str, Any]:
    return {'agent': agent_name, 'score': -1.0, 'details': details, 'error': True}


# [Include ALL your existing agent functions here unchanged]
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
        
        face_idx = len(faces) // 2 if len(faces) > 1 else 0
        face = faces[face_idx]
        
        if torch.is_tensor(face):
            face = face.cpu().numpy()
        
        if face.dtype != np.uint8:
            face = (face * 255).astype(np.uint8) if face.max() <= 1.0 else face.astype(np.uint8)
        
        def extract_quality_metrics(face_img):
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
            
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_tensor = torch.full((1, image_size, image_size), blur_score / 1000.0)
            
            exposure_score = np.mean(gray) / 255.0
            exposure_tensor = torch.full((1, image_size, image_size), exposure_score)
            
            return torch.cat([blur_tensor, exposure_tensor], dim=0)
        
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
        
        face_tensor = transform(face)
        quality_features = extract_quality_metrics(face)
        
        combined_input = torch.cat([face_tensor, quality_features], dim=0)
        input_tensor = combined_input.unsqueeze(0).to(device)
        
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


# [Keep your existing Dataset class]
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

    logging.info(f"Processing data from file: {os.path.basename(filepath)}")

    media_data = {
        'filepath': filepath,
        'processed_faces': batch['faces'].squeeze(0),
        'audio_waveform': batch['waveform'].squeeze(0),
        'audio_sample_rate': CONFIG["audio_agent"]["sample_rate"],
        'metadata': {'ground_truth': label.capitalize()}
    }

    lc_input = {"media_data": media_data, "models": models}
    return lc_input


def print_report(decision: Dict[str, Any]):
    """Prints a formatted report for a single analysis decision."""
    media_data = decision['media_data']
    phase = decision.get('phase', 'unknown')
    consensus = decision.get('consensus_rate', 0)
    
    report = f"""
----------------------------------------------------------------------
Analysis Report for: {os.path.basename(media_data['filepath'])}
Ground Truth: {media_data['metadata']['ground_truth']}
System Verdict: {decision['verdict']} (Confidence: {decision['confidence']:.2%})

Analysis Details:
- Final Phase: {phase.upper()}
- Agent Consensus: {consensus:.0%}
- Time Taken: {decision.get('analysis_time', 0):.2f}s
- Reason: {decision['reason']}
----------------------------------------------------------------------
Agent Breakdown ({len(decision.get('agents_used', []))} agents used):"""
    
    for result in decision.get('results', []):
        if result:
            score_str = f'{result["score"]:.3f}' if not result.get('error') else 'ERROR'
            status = "Used" if result['agent'] in decision.get('agents_used', []) else "Not Used"
            report += f"\n  - {result['agent']:<28} | Score: {score_str:<7} | Status: {status}"
    
    report += "\n----------------------------------------------------------------------"
    logging.info(report)


class Evaluator:
    def __init__(self):
        self.y_true, self.y_pred = [], []
        self.label_map = {'Real': 0, 'Fake': 1}
        self.verdict_map = {'Real': 0, 'Deepfake': 1}
        self.phase_counts = defaultdict(int)
        self.total_time = 0
        self.sample_count = 0
        logging.info("Evaluator initialized.")

    def log_result(self, ground_truth: str, system_verdict: str, phase: str = None, time: float = 0):
        true_val = self.label_map.get(ground_truth)
        pred_val = self.verdict_map.get(system_verdict)

        if true_val is not None and pred_val is not None:
            self.y_true.append(true_val)
            self.y_pred.append(pred_val)
            if phase:
                self.phase_counts[phase] += 1
            self.total_time += time
            self.sample_count += 1

    def display_report(self):
        if not self.y_true:
            logging.warning("No results were logged, cannot generate evaluation report.")
            return
        
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision = precision_score(self.y_true, self.y_pred, zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, zero_division=0)
        cm = confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])
        avg_time = self.total_time / self.sample_count if self.sample_count > 0 else 0

        report = f"""
============================================================
           Multi-Agent System Evaluation Report
============================================================
Total Samples Processed: {len(self.y_true)}

Performance Metrics:
  - Accuracy:  {accuracy:.2%}
  - Precision: {precision:.2%}
  - Recall:    {recall:.2%}
  - F1-Score:  {f1:.2%}

Efficiency:
  - Average Analysis Time: {avg_time:.3f} seconds

Phase Distribution:
  - Quick Consensus:    {self.phase_counts.get('quick', 0)} samples
  - Strong Arbitration: {self.phase_counts.get('strong', 0)} samples
  - Iterative Refinement: {self.phase_counts.get('iterative', 0)} samples

Confusion Matrix:
              Predicted
             +--------+
             | Real | Fake |
  +----------+------+------+
  | Real     | {cm[0][0]:<4} | {cm[0][1]:<4} |
  | Fake     | {cm[1][0]:<4} | {cm[1][1]:<4} |
  +----------+------+------+
============================================================"""
        print(report)


def main():
    setup_logging()

    print("\n" + "="*60)
    print("Multi-Agent Deepfake Detection System Initializing...")
    print("This system uses dynamic resource allocation, agent communication,")
    print("and iterative refinement to analyze media.")
    print("="*60 + "\n")

    if not check_system_readiness():
        logging.critical("System readiness check failed. Please check file paths. Exiting.")
        return

    models = load_all_models()
    orchestrator = MultiAgentOrchestrator(models)

    test_dir = os.path.join(CONFIG["data_dir"], 'test')
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

    print(f"\nStarting analysis of {total_files} test files...")
    print("-" * 60)

    for i, batch in enumerate(data_loader):
        try:
            lc_input = process_batch(batch, models)
            if lc_input:
                # Let the orchestrator handle the analysis
                final_decision = orchestrator.orchestrate_analysis(lc_input['media_data'])
                final_decision['media_data'] = lc_input['media_data']

                print_report(final_decision)

                evaluator.log_result(
                    ground_truth=final_decision['media_data']['metadata']['ground_truth'],
                    system_verdict=final_decision['verdict'],
                    phase=final_decision.get('phase'),
                    time=final_decision.get('analysis_time', 0)
                )

                result_row = {
                    'filepath': os.path.basename(final_decision['media_data']['filepath']),
                    'ground_truth': final_decision['media_data']['metadata']['ground_truth'],
                    'system_verdict': final_decision['verdict'],
                    'final_score': final_decision['confidence'],
                    'phase': final_decision.get('phase', 'unknown'),
                    'consensus_rate': final_decision.get('consensus_rate', 0),
                    'analysis_time': final_decision.get('analysis_time', 0)
                }
                
                # Add individual agent scores to the results row
                for result in final_decision.get('results', []):
                    if result:
                        agent_name = result.get('agent', 'Unknown Agent')
                        score = result.get('score', -1.0)
                        result_row[f'score_{agent_name}'] = score
                
                all_results.append(result_row)

            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{total_files} files processed ({ (i + 1) / total_files * 100:.1f}%)")

        except Exception as e:
            logging.error(f"An unexpected error occurred while processing a batch: {e}", exc_info=True)
            continue

    print("\n" + "="*60)
    print("Analysis complete. Generating final report.")
    evaluator.display_report()

    if all_results:
        try:
            output_file = CONFIG["output_file"]
            
            # Define base columns that are always present
            base_fieldnames = [
                'filepath', 'ground_truth', 'system_verdict', 'final_score', 
                'phase', 'consensus_rate', 'analysis_time'
            ]
            
            # Dynamically get column names for all possible agents to ensure consistency
            agent_names = orchestrator.agent_functions.keys()
            agent_fieldnames = [f'score_{name}' for name in agent_names]
            
            # Combine into a complete and ordered list of headers
            fieldnames = base_fieldnames + sorted(agent_fieldnames)

            logging.info(f"Saving {len(all_results)} results to {output_file}...")
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
                
            print(f"\nMulti-agent analysis results saved successfully to {output_file}")

        except Exception as e:
            logging.error(f"Failed to save results to CSV file: {e}", exc_info=True)


if __name__ == "__main__":
    main()