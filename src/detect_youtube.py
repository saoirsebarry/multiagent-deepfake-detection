# -*- coding: utf-8 -*-
"""
Enhanced multi-agent system for deepfake detection with 5 agents and XAI visualizations
WITH EXTENSIVE DEBUG LOGGING FOR FILE PROCESSING
Includes: Visual (Spatial), Audio (FreqNet), Audio Forensics (ECAPA), 
Cross-Modal (Lip-Sync), and Face Quality agents with Grad-CAM and SHAP support
"""

import traceback 
import os
import logging
import traceback
import warnings
from typing import Dict, Any, List, Optional, Tuple
import cv2
import time
import shutil
import glob
import dlib
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import librosa
from PIL import Image
from torchvision import transforms, models
from speechbrain.pretrained import EncoderClassifier
import shap

# model architectures
from agents.visual_xception import XceptionDeepfakeDetector
from agents.cross_modal_lipsync import CrossModal_CNN_LSTM
from agents.audio_freqnet import FreqNet
from agents.audio_forensics_ecapa import OptimizedLightweightForensics, FastAudioFeatureExtractor

from xai_utils import (
    GradCAM, create_visual_overlay, create_audio_overlay, generate_cross_attention_viz
)
# fill in your api key
#GROQ_API_KEY = 
#GOOGLE_API_KEY = 

CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir": "data/face_audio_dataset_youtube",
    "model_files": {
        "spatial": "checkpoints/xception/polyglotfake_xception_best_unbal_all_faceaug.pth",
        "audio": "checkpoints/freqnet/freqnet_model_all_unbalanced_improved.pth",
        "audio_forensics": "checkpoints/ecapa_forensic_head/audio_forensics_model_finetuned_best.pth",
        "cross_modal": "checkpoints/cross_modal/lip_sync_model_crossattention.pth",
        "face_quality": "checkpoints/biometric/fine_tuning/best_model.pth",
    },
    "audio_forensics_stats_path": "checkpoints/ecapa_forensic_head/training_stats.npz",
    "gradcam_output_dir": "multiagent_xai_results_5agents_yt",
    "visual_agent": {"image_size": 299},
    "audio_agent": {"n_mels": 224, "sample_rate": 16000, "clip_duration_s": 5, "hop_length": 512},
    "audio_forensics_agent": {
        "sample_rate": 16000, 
        "duration": 6, 
        "embedding_dim": 192, 
        "num_forensic_features": 11, 
        "window_size": 2.0, 
        "hop_size": 1.0
    },
    "cross_modal_agent": {"max_faces": 20, "image_size": 224, "audio_target_length": 313, "sample_rate": 16000, "n_fft": 2048, "hop_length": 512, "n_mels": 128, "video_fps": 25, "frame_stride": 10},
    "face_quality_agent": {"image_size": 299, "normalize": True},
    "debug_mode": True,
    "target_file": None,
    "decision_engine": {
        "threshold": 0.37,
    },
    "multi_agent": {
        "confidence_threshold": 0.7,
        "disagreement_threshold": 0.01,
        "second_wave_agents": ["Visual (Spatial)", "Audio (FreqNet)"],
        "first_wave_agents": ["Face Quality","Audio Forensics (ECAPA)" ,"Cross-Modal (Lip-Sync)"],
        "max_iterations": 3,
        "convergence_threshold": 0.05,
    }
}


import google.generativeai as genai
from groq import Groq
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from collections import defaultdict


class AgentCommunicationHub:
    """Central hub for agent communication and coordination"""
    
    def __init__(self):
        self.message_queue = defaultdict(list)
        self.analysis_history = []
        self.agent_states = {}
        
    def record_analysis(self, agent_name: str, result: Dict[str, Any]):
        """Record an agent's analysis result"""
        self.analysis_history.append({
            'agent': agent_name,
            'result': result,
            'timestamp': time.time()
        })
        self.agent_states[agent_name] = result
        
    def get_consensus(self) -> Tuple[float, bool]:
        """Calculate consensus among agents"""
        if not self.agent_states:
            return 0.5, False
            
        scores = [r['score'] for r in self.agent_states.values() 
                 if not r.get('error') and r.get('score', -1) >= 0]
        if not scores:
            return 0.5, False
            
        verdicts = ['fake' if s > 0.5 else 'real' for s in scores]
        consensus_verdict = max(set(verdicts), key=verdicts.count)
        agreement_rate = verdicts.count(consensus_verdict) / len(verdicts)
        
        score_variance = np.var(scores) if len(scores) > 1 else 0
        has_consensus = agreement_rate >= 0.8 and score_variance < 0.1
        
        return agreement_rate, has_consensus
    
    def identify_conflicts(self) -> List[Tuple[str, str]]:
        """Identify pairs of agents with conflicting results"""
        conflicts = []
        agents = list(self.agent_states.keys())
        
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                score1 = self.agent_states[agents[i]].get('score', 0.5)
                score2 = self.agent_states[agents[j]].get('score', 0.5)
                
                if (score1 > 0.5) != (score2 > 0.5):
                    conflicts.append((agents[i], agents[j]))
                    
        return conflicts

class MultiAgentOrchestrator:
    """Orchestrates two-phase multi-agent analysis with XAI visualization"""
    
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.comm_hub = AgentCommunicationHub()
        self.ma_config = CONFIG["multi_agent"]
        
        # Agent functions 
        self.agent_functions = {
            "Visual (Spatial)": run_visual_analysis,
            "Audio (FreqNet)": run_audio_analysis,
            "Audio Forensics (ECAPA)": run_audio_forensics_analysis,
            "Cross-Modal (Lip-Sync)": run_cross_modal_analysis,
            "Face Quality": run_face_quality_analysis,
        }
        
    def orchestrate_analysis(self, media_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate two-phase multi-agent analysis with XAI"""
        start_time = time.time()

        logging.info("="*60)
        logging.info("PHASE 1: Deploying First Wave Agents")
        logging.info("="*60)
        first_wave_results = self._run_first_wave(media_data)
        
        # Check for disagreement
        disagreement = self._calculate_disagreement(first_wave_results)
        logging.info(f"First wave disagreement level: {disagreement:.3f}")
        
        all_results = first_wave_results.copy()
        
        # Phase 2: Deploy additional agents if disagreement > threshold
        if disagreement > self.ma_config['disagreement_threshold']:
            logging.info("="*60)
            logging.info("⚡ PHASE 2: High disagreement detected - Deploying Second Wave Agents")
            logging.info("="*60)
            second_wave_results = self._run_second_wave(media_data)
            all_results.update(second_wave_results)
        else:
            logging.info(f"Low disagreement ({disagreement:.3f}) - second wave not needed")
        
        # Update communication hub and finalize decision
        self._update_communication_hub(all_results)
        decision = self._finalize_decision(all_results, start_time=start_time)
        
        return decision
    
    def _run_first_wave(self, media_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run first wave agents with XAI visualizations"""
        results = {}
        for agent_name in self.ma_config['first_wave_agents']:
            if agent_name in self.agent_functions:
                try:
                    logging.info(f"Running {agent_name} with XAI visualization...")
                    result = self.agent_functions[agent_name](media_data, self.models)
                    results[agent_name] = result
                    self.comm_hub.record_analysis(agent_name, result)
                    
                    # Log XAI visualization path if generated
                    if result.get('visualization_path'):
                        logging.info(f"XAI visualization saved: {result['visualization_path']}")
                        
                except Exception as e:
                    logging.error(f"First wave agent {agent_name} failed: {e}")
                    results[agent_name] = create_error_result(agent_name, str(e))
        return results
    
    def _run_second_wave(self, media_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy second wave agents with XAI analysis"""
        results = {}
        for agent_name in self.ma_config['second_wave_agents']:
            if agent_name in self.agent_functions:
                try:
                    logging.info(f"Deploying second wave agent: {agent_name}")
                    result = self.agent_functions[agent_name](media_data, self.models)
                    results[agent_name] = result
                    self.comm_hub.record_analysis(agent_name, result)
                    
                    if result.get('visualization_path'):
                        logging.info(f"XAI visualization saved: {result['visualization_path']}")
                        
                except Exception as e:
                    logging.error(f"Second wave agent {agent_name} failed: {e}")
                    results[agent_name] = create_error_result(agent_name, str(e))
        return results
    
    def _calculate_disagreement(self, results: Dict[str, Any]) -> float:
        """Calculate disagreement level among agents (0-1 scale)"""
        valid_scores = [r['score'] for r in results.values() 
                       if not r.get('error') and r.get('score', -1) >= 0]
        
        if len(valid_scores) < 2:
            return 0.0  
        
        # Higher std dev = more disagreement
        scores_array = np.array(valid_scores)
        disagreement = np.std(scores_array)
        

        verdicts = ['fake' if s > 0.5 else 'real' for s in valid_scores]
        if len(set(verdicts)) > 1:
            # ensure disagreement is at least 0.3
            disagreement = max(disagreement, 0.3)
        
        return float(disagreement)
    
    def _finalize_decision(self, results: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Generate final decision with equal weighting"""
        valid_results = {k: v for k, v in results.items() 
                        if not v.get('error') and v.get('score', -1) >= 0}
        
        if not valid_results:
            return {
                'verdict': 'Uncertain',
                'confidence': 0.0,
                'reason': 'All agents failed',
                'results': list(results.values()),
                'analysis_time': time.time() - start_time,
                'visualizations': []
            }
        
        scores = [result['score'] for result in valid_results.values()]
        final_score = np.mean(scores)
        
        # visualizations
        visualizations = []
        for agent_name, result in valid_results.items():
            if result.get('visualization_path'):
                visualizations.append({
                    'agent': agent_name,
                    'path': result['visualization_path']
                })
        
        # verdict
        threshold = CONFIG["decision_engine"]["threshold"]
        verdict = "DEEPFAKE" if final_score >= threshold else "REAL"

        consensus_rate, has_consensus = self.comm_hub.get_consensus()
        
        # get reason
        num_agents = len(valid_results)
        phase_info = "first wave only" if num_agents <= 3 else "both waves"
        disagreement = self._calculate_disagreement(results)
        
        reason = f"Based on {num_agents} agents ({phase_info}). "
        reason += f"Average score: {final_score:.3f}, Disagreement: {disagreement:.3f}"
        

        if visualizations:
            logging.info("="*60)
            logging.info("XAI VISUALIZATIONS GENERATED:")
            for viz in visualizations:
                logging.info(f"  • {viz['agent']}: {viz['path']}")
            logging.info("="*60)
        
        return {
            'verdict': verdict,
            'confidence': final_score,
            'reason': reason,
            'results': list(results.values()),
            'consensus_rate': consensus_rate,
            'agents_used': list(valid_results.keys()),
            'analysis_time': time.time() - start_time,
            'disagreement_level': disagreement,
            'visualizations': visualizations
        }
    
    def _update_communication_hub(self, results: Dict[str, Any]):
        """Update communication hub with results"""
        for agent_name, result in results.items():
            if result and not result.get('error'):
                self.comm_hub.agent_states[agent_name] = result

# report generation 
def generate_llama_report(decision: Dict[str, Any]):
    """Generates detailed forensic report with dynamic weighting information"""
    media_data = decision.get('media_data', {})
    filename = os.path.basename(media_data.get('filepath', 'unknown.npz'))
    
    metadata = {}
    if 'filepath' in media_data:
        try:
            data = np.load(media_data['filepath'], allow_pickle=True)
            if 'metadata' in data:
                metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else {}
            elif 'label' in data:
                label = data.get('label', ["Unknown"])[0]
                metadata = {'ground_truth': label.capitalize()}
        except:
            pass
    
    ground_truth = metadata.get('ground_truth', 'Unknown')
    verdict = decision['verdict']
    confidence = decision['confidence']
    results = decision['results']
    threshold = CONFIG['decision_engine']['threshold']
    dynamic_weights = decision.get('dynamic_weights', {})
    phase = decision.get('phase', 'unknown')
    consensus = decision.get('consensus_rate', 0)


    agent_summary = ""
    core_agents = ["Visual (Spatial)", "Audio (FreqNet)", "Audio Forensics (ECAPA)", 
                   "Cross-Modal (Lip-Sync)", "Face Quality"]
    
    for agent_name in core_agents:
        result = next((r for r in results if r['agent'] == agent_name), None)
        if result:
            if not result.get('error'):
                score = result['score']
                weight = dynamic_weights.get(agent_name, 0)
                contribution = score * weight
                status = f"Score: {score:.4f} | Weight: {weight:.2%} | Contribution: {contribution:.4f}"
            else:
                status = f"ERROR ({result.get('details', 'Unknown error')})"
            agent_summary += f"- **{agent_name}**: {status}\n"

    # semantic context
    visual_context_result = next((r for r in results if r['agent'] == 'Visual Context'), None)
    transcription_result = next((r for r in results if r['agent'] == 'Audio Transcription'), None)
    
    visual_description = "Not performed"
    if visual_context_result:
        visual_description = visual_context_result.get('description', 'Analysis failed')
    
    audio_transcript = "Not performed"
    if transcription_result:
        audio_transcript = transcription_result.get('transcript', 'Transcription failed')

    prompt = f"""
**Role:** You are a Senior Digital Forensics Expert specializing in deepfake detection using multi-agent AI systems.

**Task:** Write a comprehensive forensic report documenting the dynamic multi-agent analysis of a media file.

---
### CASE DETAILS ###
- **File:** `{filename}`
- **Ground Truth:** `{ground_truth}`
- **Analysis Phase:** `{phase.upper()}`
- **Agent Consensus:** `{consensus:.1%}`
- **Processing Time:** `{decision.get('analysis_time', 0):.2f} seconds`

### DYNAMIC MULTI-AGENT ANALYSIS ###
**Final Verdict:** `{verdict}`
**Confidence Score:** `{confidence:.4f}` (Threshold: {threshold})

**Agent Contributions (with Dynamic Weights):**
{agent_summary}

**Semantic Context:**
- Visual: "{visual_description}"
- Audio: "{audio_transcript}"

### XAI Visualizations Generated ###
{len(decision.get('visualizations', []))} visualization files created for explainability

---
### REPORT REQUIREMENTS ###

1. **Executive Summary**
   - State the verdict and confidence clearly
   - Explain how the dynamic weighting system reached this conclusion
   - Mention which phase was sufficient (quick/strong/iterative)

2. **Dynamic Weight Analysis**
   - Explain why certain agents received higher weights
   - Discuss how agent agreement/disagreement affected the final decision
   - Note any strong signals that influenced the outcome

3. **Contextual Assessment**
   - If semantic analysis was performed, assess plausibility
   - Note any inconsistencies between visual and audio content

4. **Verification**
   - Compare verdict with ground truth
   - State whether the system was CORRECT or INCORRECT
   - Suggest areas for improvement if incorrect

Write a professional, detailed report that demonstrates the sophistication of the dynamic multi-agent approach.
"""

    try:
        if not GROQ_API_KEY or "YOUR_GROQ_API" in GROQ_API_KEY:
            raise ValueError("Groq API Key not configured")
        
        client = Groq(api_key=GROQ_API_KEY)
        logging.info(f"Generating enhanced forensic report")
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=1500
        )
        
        print("\n" + "="*80)
        print(f" DYNAMIC MULTI-AGENT FORENSIC ANALYSIS REPORT")
        print(f" File: {filename.upper()}")
        print("="*80)
        print(chat_completion.choices[0].message.content)
        print("="*80)
        
        
    except Exception as e:
        logging.warning(f"LLM report generation failed: {e}. Using enhanced fallback.")
        
        # Enhanced fallback report
        print("\n" + "="*80)
        print(f" ANALYSIS REPORT: {filename.upper()}")
        print("="*80)
        print(f"\nVERDICT: {verdict}")
        print(f"CONFIDENCE: {confidence:.2%}")
        print(f"THRESHOLD: {threshold}")
        print(f"GROUND TRUTH: {ground_truth}")
        print(f"ASSESSMENT: {'CORRECT' if verdict == ground_truth.upper() else 'INCORRECT'}")
        print(f"\nANALYSIS PHASE: {phase.upper()}")
        print(f"CONSENSUS RATE: {consensus:.1%}")
        print(f"PROCESSING TIME: {decision.get('analysis_time', 0):.2f}s")
        
        print("\n--- Dynamic Agent Analysis ---")
        for r in results:
            if not r.get('error'):
                agent = r['agent']
                score = r.get('score', -1)
                weight = dynamic_weights.get(agent, 0)
                if score >= 0:
                    print(f"  ✓ {agent:<25}: Score={score:.3f}, Weight={weight:.1%}")
            else:
                print(f"{r['agent']:<25}: ERROR")
        
        if decision.get('visualizations'):
            print(f"\n--- XAI Visualizations ---")
            for viz in decision['visualizations']:
                print(f"{viz['agent']}: {viz['path']}")
        
        print("="*80 + "\n")


from langchain_core.runnables import RunnableLambda, RunnableParallel

def load_all_models() -> Dict[str, Any]:
    models = {}
    device = CONFIG['device']
    logging.info(f"Using device: {device}")
    
    # Visual, Audio, Cross-Modal, Face Quality Models
    try:
        visual_model = XceptionDeepfakeDetector(num_classes=1).to(device)
        checkpoint = torch.load(CONFIG["model_files"]["spatial"], map_location=device)
        visual_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        models['spatial'] = visual_model.eval()
    except Exception as e: logging.error(f"Visual model failed: {e}")

    try:
        audio_model = FreqNet(num_classes=1).to(device)
        audio_model.load_state_dict(torch.load(CONFIG["model_files"]["audio"], map_location=device))
        models['audio'] = audio_model.eval()
    except Exception as e: logging.error(f"Audio model failed: {e}")

    try:
        lip_sync_model = CrossModal_CNN_LSTM().to(device)
        lip_sync_model.load_state_dict(torch.load(CONFIG["model_files"]["cross_modal"], map_location=device))
        models['cross_modal'] = lip_sync_model.eval()
    except Exception as e: logging.error(f"Cross-modal model failed: {e}")
    
    try:
        face_quality_model = FaceQualityNet(num_classes=1).to(device)
        checkpoint = torch.load(CONFIG["model_files"]["face_quality"], map_location=device)
        face_quality_model.load_state_dict(checkpoint['model_state_dict'])
        models['face_quality'] = face_quality_model.eval()
    except Exception as e: logging.error(f"Face Quality model failed: {e}")
    

    try:
        # trained forensics model
        cfg_af = CONFIG['audio_forensics_agent']
        audio_forensics_model = OptimizedLightweightForensics(
            embedding_dim=cfg_af['embedding_dim'],
            num_forensic_features=cfg_af['num_forensic_features']
        ).to(device).eval()
        audio_forensics_model.load_state_dict(torch.load(CONFIG["model_files"]["audio_forensics"], map_location=device))
        models['audio_forensics'] = audio_forensics_model

        # speaker encoder
        models['speaker_encoder'] = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="checkpoints/speechbrain_cache",
            run_opts={"device": device}
        ).eval()


        stats = np.load(CONFIG['audio_forensics_stats_path'])
        models['audio_forensics_stats'] = {
            'mean': torch.tensor(stats['mean'], dtype=torch.float32, device=device),
            'std': torch.tensor(stats['std'], dtype=torch.float32, device=device)
        }
        

        def f(x):
            input_tensor = torch.tensor(x, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = models['audio_forensics'](input_tensor)
                scores = torch.sigmoid(logits)
            return scores.cpu().numpy()

        # Create a small background dataset for the explainer (e.g., 100 samples of random noise)
        background_data = np.zeros((100, cfg_af['embedding_dim'] + cfg_af['num_forensic_features']))
        models['shap_explainer'] = shap.KernelExplainer(f, background_data)
        
        logging.info("Audio Forensics pipeline (Model, Encoder, Stats, SHAP) loaded.")

    except Exception as e:
        logging.error(f"Audio Forensics pipeline failed: {e}\n{traceback.format_exc()}")
        
    return models

# Audio Forensics Agent 
def run_audio_forensics_analysis(media_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes audio using the full forensics pipeline and generates a SHAP plot.
    """
    agent_name = "Audio Forensics (ECAPA)"
    required_keys = ['audio_forensics', 'speaker_encoder', 'audio_forensics_stats', 'shap_explainer']
    if any(key not in models for key in required_keys):
        return create_error_result(agent_name, "One or more model components failed to load.")

    model = models['audio_forensics']
    speaker_encoder = models['speaker_encoder']
    stats = models['audio_forensics_stats']
    explainer = models['shap_explainer']
    
    waveform = media_data.get('audio_waveform')
    if not isinstance(waveform, np.ndarray) or waveform.size < 400:
        return create_error_result(agent_name, "Insufficient audio data.")
    
    try:
        cfg = CONFIG["audio_forensics_agent"]
        sr = cfg['sample_rate']
        device = CONFIG['device']
        feature_extractor = FastAudioFeatureExtractor()
        
        # preprocess audio
        target_length = sr * cfg['duration']
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)), 'constant')
        
        # extract all features
        prosody = feature_extractor.extract_fast_prosody(waveform, sr)
        artifacts = feature_extractor.extract_fast_artifacts(waveform, sr)
        
        with torch.no_grad(), autocast():
            wav_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
            embedding = speaker_encoder.encode_batch(wav_tensor).squeeze()
            
            window_size = int(cfg['window_size'] * sr)
            chunks = [waveform[i:i + window_size] for i in [0, len(waveform)//2 - window_size//2, len(waveform) - window_size] if i >= 0 and i + window_size <= len(waveform)]
            if len(chunks) >= 2:
                chunk_tensor = torch.tensor(np.array(chunks), dtype=torch.float32).to(device)
                chunk_embeddings = speaker_encoder.encode_batch(chunk_tensor)
                distances = torch.norm(chunk_embeddings[:-1] - chunk_embeddings[1:], dim=2).mean(dim=1)
                temporal = torch.tensor([torch.mean(distances), torch.std(distances)], device=device)
                embedding_var = torch.std(chunk_embeddings, dim=0).mean()
            else:
                temporal = torch.zeros(2, device=device)
                embedding_var = torch.tensor(0.0, device=device)

        # normalize features
        all_features = torch.cat([
            embedding,
            torch.tensor(prosody, device=device),
            torch.tensor(artifacts, device=device),
            temporal,
            embedding_var.unsqueeze(0)
        ]).float()
        
        normalized_features = (all_features - stats['mean']) / (stats['std'] + 1e-6)
        input_tensor = normalized_features.unsqueeze(0)

        # model prediction
        with torch.no_grad():
            score = torch.sigmoid(model(input_tensor)).squeeze().item()

        # shap
        base_filename = os.path.basename(media_data['filepath']).replace('.npz', '')
        viz_path = os.path.join(CONFIG['gradcam_output_dir'], f"shap_forensics_{base_filename}.png")
        
        shap_values = explainer.shap_values(input_tensor.cpu().numpy())
        raw_values = shap_values[0].flatten()
        raw_data = input_tensor.cpu().numpy()[0]

        original_feature_names = [f'emb_{i}' for i in range(cfg['embedding_dim'])] + \
                                 ['pitch_mean', 'pitch_std', 'pitch_range', 'energy_mean', 'energy_std'] + \
                                 ['hf_ratio', 'spectral_flux', 'spectral_centroid'] + \
                                 ['temp_mean_dist', 'temp_std_dist', 'embedding_var']


        final_shap_values = []
        final_data_values = []
        final_feature_names = []

        # ECAPA-TDNN embedding features
        embedding_indices = list(range(cfg['embedding_dim']))
        final_shap_values.append(raw_values[embedding_indices].sum())
        final_data_values.append(raw_data[embedding_indices].mean())
        final_feature_names.append("'ECAPA-TDNN' Profile")

        # remaining forensic features individually
        for i in range(cfg['embedding_dim'], len(original_feature_names)):
            final_shap_values.append(raw_values[i])
            final_data_values.append(raw_data[i])
            final_feature_names.append(original_feature_names[i])

        # final, hybrid Explanation object
        shap_explanation = shap.Explanation(
            values=np.array(final_shap_values),
            base_values=explainer.expected_value,
            data=np.array(final_data_values),
            feature_names=final_feature_names
        )
        
        plt.figure()
        # 12 features
        shap.plots.waterfall(shap_explanation, max_display=12, show=False)

        plt.title(f"Audio Forensics SHAP Analysis (Score: {score:.2f})")
        plt.tight_layout()
        plt.savefig(viz_path, dpi=150)
        plt.close()

        return {
            'agent': agent_name, 'score': float(score),
            'details': 'Hybrid SHAP analysis complete.',
            'visualization_path': viz_path
        }
        
    except Exception as e:
        logging.error(f"[{agent_name}] failed: {e}\n{traceback.format_exc()}")
        return create_error_result(agent_name, str(e))

def create_error_result(agent_name: str, details: str) -> Dict[str, Any]:
    """Standardized error dictionary for agents."""
    return {'agent': agent_name, 'score': -1.0, 'details': details, 'error': True, 'visualization_path': None}


def setup_debug_logging():
    debug_log_file = "debug_file_processing.log"
    logging.basicConfig(
        level=logging.DEBUG if CONFIG["debug_mode"] else logging.INFO,
        format="[%(asctime)s] [%(levelname)-8s] [%(funcName)-20s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(debug_log_file, mode='w'), logging.StreamHandler()]
    )
    warnings.filterwarnings("ignore")
    logging.getLogger("speechbrain").setLevel(logging.WARNING)
    logging.getLogger("shap").setLevel(logging.WARNING)
    logging.getLogger("speechbrain").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")
    output_dir = CONFIG['gradcam_output_dir']
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    logging.info(f"Clean XAI output directory created: {output_dir}")


def debug_file_discovery():
    test_dir = os.path.join(CONFIG["data_dir"], 'test')
    if not os.path.exists(test_dir):
        logging.error(f" Test directory does not exist: {test_dir}")
        return []
    final_files = glob.glob(os.path.join(test_dir, "**", "*.npz"), recursive=True)
    logging.info(f"FINAL: {len(final_files)} unique NPZ files found")
    return final_files

def debug_file_selection(test_files):
    if not test_files:
        logging.error("No files available for selection!")
        return None
    return test_files[0]

def debug_file_loading(filepath):
    try:
        data = np.load(filepath, allow_pickle=True)
        return {
            'filepath': filepath,
            'processed_faces': data.get('faces', np.array([])),
            'audio_waveform': data.get('waveform', np.array([])),
            'audio_sample_rate': CONFIG['audio_agent']['sample_rate']
        }
    except Exception as e:
        logging.error(f"Failed to load NPZ file: {e}\n{traceback.format_exc()}")
        return None

# model architecture
class FaceQualityNet(nn.Module):
    def __init__(self, num_classes: int = 1):
        super(FaceQualityNet, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=False)
        orig_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(5, orig_conv.out_channels, kernel_size=orig_conv.kernel_size, stride=orig_conv.stride, padding=orig_conv.padding, bias=False)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.quality_head = nn.Sequential(nn.Linear(num_features, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5), nn.Linear(256, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3))
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, num_classes))
    def forward(self, x):
        features = self.backbone(x)
        quality_features = self.quality_head(features)
        output = self.classifier(quality_features)
        return torch.sigmoid(output)



def run_visual_analysis(media_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes video frames and generates a multi-frame Grad-CAM plot.
    This version plots 10 frames and includes the original image for comparison.
    """
    agent_name = "Visual (Spatial)"
    model = models.get('spatial')
    if model is None: return create_error_result(agent_name, "Model not loaded.")

    faces = media_data.get('processed_faces')
    if not isinstance(faces, np.ndarray) or faces.size == 0:
        return create_error_result(agent_name, "No processed faces found in media_data.")

    cfg = CONFIG['visual_agent']
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg['image_size'], cfg['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    try:
        # convert BGR to RGB
        first_face_rgb = cv2.cvtColor(faces[0], cv2.COLOR_BGR2RGB)
        first_face_tensor = transform(first_face_rgb).unsqueeze(0).to(CONFIG['device'])
        
        with torch.no_grad():
            prediction_logit = model(first_face_tensor)
            score = torch.sigmoid(prediction_logit).item()
    except Exception as e:
        return create_error_result(agent_name, f"Inference failed: {e}")

    viz_path = None
    try:
        base_filename = os.path.basename(media_data['filepath']).replace('.npz', '')
        viz_path = os.path.join(CONFIG['gradcam_output_dir'], f"gradcam_visual_{base_filename}.png")

        target_layer = None
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = module
                logging.info(f"[{agent_name}] Automatically selected Grad-CAM target layer: {name}")
                break
        
        if not target_layer: raise RuntimeError("Could not find a Conv2d layer for Grad-CAM.")
        
        cam_generator = GradCAM(model, target_layer)

        MAX_FRAMES_TO_PLOT = min(10, len(faces))
        
        fig, axs = plt.subplots(1, 1 + MAX_FRAMES_TO_PLOT, figsize=((1 + MAX_FRAMES_TO_PLOT) * 4, 4))

        # BGR to RGB for display
        first_face_display = cv2.cvtColor(faces[0], cv2.COLOR_BGR2RGB)
        axs[0].imshow(first_face_display)
        axs[0].set_title("Original")
        axs[0].axis('off')

        for i in range(MAX_FRAMES_TO_PLOT):
            frame_rgb = cv2.cvtColor(faces[i], cv2.COLOR_BGR2RGB)
            input_tensor = transform(frame_rgb).unsqueeze(0).to(CONFIG['device'])
            
            heatmap = cam_generator(input_tensor)
            overlay = create_visual_overlay(frame_rgb, heatmap)
            
            axs[i + 1].imshow(overlay)
            axs[i + 1].set_title(f"Frame {i+1}")
            axs[i + 1].axis('off')
        
        fig.suptitle(f"Visual Agent Grad-CAM (Score: {score:.3f})", fontsize=16, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_path, dpi=150)
        plt.close(fig)

    except Exception as e:
        logging.error(f"[{agent_name}] Grad-CAM plot generation FAILED. Error: {e}\n{traceback.format_exc()}")
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        viz_path = None 
    finally:
        if 'cam_generator' in locals(): cam_generator.remove_hooks()

    return {'agent': agent_name, 'score': float(score), 'details': 'PyTorch Xception analysis.', 'visualization_path': viz_path}

    
def run_audio_analysis(media_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    agent_name = "Audio (FreqNet)"
    model = models.get('audio')
    waveform = media_data.get('audio_waveform')
    if model is None or not isinstance(waveform, np.ndarray) or waveform.size < 400:
        return create_error_result(agent_name, "Model or data unavailable.")
    
    score, viz_path = -1.0, None
    try:
        cfg = CONFIG['audio_agent']
        sr = cfg['sample_rate']
        target_length = sr * cfg['clip_duration_s']
        y = waveform[:target_length] if len(waveform) > target_length else np.pad(waveform, (0, target_length - len(waveform)))
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=cfg["n_mels"], hop_length=cfg["hop_length"])
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        norm_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
        input_tensor = torch.tensor(norm_spec, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(CONFIG['device'])
        with torch.no_grad(): score = torch.sigmoid(model(input_tensor)).item()

        target_layer = next((m for m in reversed(list(model.modules())) if isinstance(m, nn.Conv2d)), None)
        base_filename = os.path.basename(media_data['filepath']).replace('.npz', '')
        viz_path = os.path.join(CONFIG['gradcam_output_dir'], f"gradcam_audio_{base_filename}.png")
        cam_generator = GradCAM(model, target_layer)
        heatmap = cam_generator(input_tensor)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].imshow(log_mel_spec, aspect='auto', origin='lower', cmap='viridis'); axes[0, 0].set_title('Original Mel Spectrogram')
        heatmap_resized = cv2.resize(heatmap, (log_mel_spec.shape[1], log_mel_spec.shape[0]))
        axes[0, 1].imshow(heatmap_resized, aspect='auto', origin='lower', cmap='hot'); axes[0, 1].set_title('Grad-CAM Attention')
        overlay_rgb = cv2.cvtColor(create_audio_overlay(log_mel_spec, heatmap_resized), cv2.COLOR_BGR2RGB)
        axes[1, 0].imshow(overlay_rgb, aspect='auto', origin='lower'); axes[1, 0].set_title('Grad-CAM Overlay')
        axes[1, 1].plot(np.linspace(0, len(waveform)/sr, len(waveform)), waveform); axes[1, 1].set_title('Waveform')
        fig.suptitle(f"Audio Analysis - Score: {score:.3f}", fontsize=14); plt.tight_layout(); plt.savefig(viz_path); plt.close(fig)
        cam_generator.remove_hooks()

    except Exception as e:
        logging.error(f"[{agent_name}] failed: {e}\n{traceback.format_exc()}")
        return create_error_result(agent_name, str(e))
        
    return {'agent': agent_name, 'score': score, 'details': 'Analysis complete.', 'visualization_path': viz_path}

def run_cross_modal_analysis(media_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """
    An instrumented version of the cross-modal agent for deep debugging.
    Logs the shape, content, and statistics of all data at each processing step.
    """
    agent_name = "Cross-Modal (Lip-Sync)"


    model = models.get('cross_modal')
    faces = media_data.get('processed_faces')
    waveform = media_data.get('audio_waveform')

    if model is None or not isinstance(faces, np.ndarray) or faces.size < 100 or not isinstance(waveform, np.ndarray) or waveform.size < 4096:
        logging.error("[FAIL] Initial data is invalid or missing. Aborting.")
        return create_error_result(agent_name, "Model or required media data is unavailable.")
    

    cfg = CONFIG["cross_modal_agent"]
    visual_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg["image_size"], cfg["image_size"]), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    num_faces = min(len(faces), cfg["max_faces"])
    padded_faces = torch.zeros((cfg["max_faces"], 3, cfg["image_size"], cfg["image_size"]), dtype=torch.float32)

    for i in range(num_faces):
        padded_faces[i] = visual_transforms(faces[i])

    visual_tensor = padded_faces.unsqueeze(0).to(CONFIG['device'])
    
    # Sanity check the tensor content
    frame_means = [visual_tensor[0, i].mean().item() for i in range(num_faces)]
    if len(frame_means) > 1 and all(v == 0 for v in frame_means[1:]):
        logging.warning("  - [WARNING] All frames after the first are zero. This is a likely source of error.")

    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=cfg["sample_rate"], n_fft=cfg["n_fft"], hop_length=cfg["hop_length"], n_mels=cfg["n_mels"])
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    log_mel_spec_padded = np.zeros((cfg["n_mels"], cfg["audio_target_length"]), dtype=np.float32)
    current_len = min(log_mel_spec.shape[1], cfg["audio_target_length"])
    log_mel_spec_padded[:, :current_len] = log_mel_spec[:, :current_len]
    
    audio_tensor = torch.from_numpy(log_mel_spec_padded).unsqueeze(0).unsqueeze(0).to(CONFIG['device'])
    
    if audio_tensor.mean().item() == 0:
        logging.warning("  - [WARNING] Audio tensor is all zeros.")

    try:
        logging.info("[CHECK 4] Running model inference...")
        with torch.no_grad():
            output, cross_attention_weights = model(visual_tensor, audio_tensor)
            score = torch.softmax(output, dim=1)[0, 1].item()
        
        weights_raw = cross_attention_weights.squeeze().cpu().numpy()
        logging.info(f"Analyzing extracted attention weights (shape: {weights_raw.shape})")
        logging.info(f"  - First 10 weights: {[f'{w:.3f}' for w in weights_raw[:10]]}")
        logging.info(f"  - Stats (Min / Max / Mean / Std): {weights_raw.min():.3f} / {weights_raw.max():.3f} / {weights_raw.mean():.3f} / {weights_raw.std():.3f}")
        
        if weights_raw.argmax() == 0:
            logging.warning("  - [ISSUE DETECTED] The highest attention weight is on the VERY FIRST frame.")


        base_filename = os.path.basename(media_data['filepath']).replace('.npz', '')
        viz_path = generate_cross_attention_viz(
            cross_attention_weights,
            faces,
            base_filename,
            CONFIG['gradcam_output_dir'],
            cfg['video_fps'],
            cfg['frame_stride']
        )
        logging.info(f"  - Visualization saved to: {viz_path}")
        logging.info("="*22 + f" DEBUG FOR {agent_name} END " + "="*23 + "\n")

        return {
            'agent': agent_name,
            'score': score,
            'details': 'Debug analysis complete.',
            'visualization_path': viz_path
        }

    except Exception as e:
        logging.error(f"[{agent_name}] analysis FAILED during debug: {e}", exc_info=True)
        return create_error_result(agent_name, str(e))

import matplotlib.patches as mpatches

def run_face_quality_analysis(media_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes face quality, generating a compact 2x2 visualization with a combined
    plot for blur and exposure metrics, now including a key for the blur overlay.
    """
    agent_name = "Face Quality"
    model = models.get('face_quality')
    faces = media_data.get('processed_faces')

    if model is None or not isinstance(faces, np.ndarray) or faces.size == 0:
        return create_error_result(agent_name, "Model or processed faces are unavailable.")

    score, viz_path = -1.0, None
    try:
        cfg = CONFIG['face_quality_agent']
        image_size = cfg['image_size']
        

        face_bgr = faces[len(faces) // 2]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_model_input = blur_score / 1000.0
        exposure_model_input = np.mean(gray) / 255.0
        
        rgb_tensor = transform(face_rgb)
        blur_tensor = torch.full((1, image_size, image_size), blur_model_input)
        exposure_tensor = torch.full((1, image_size, image_size), exposure_model_input)
        input_tensor = torch.cat([rgb_tensor, blur_tensor, exposure_tensor], dim=0).unsqueeze(0).to(CONFIG['device'])
        
        with torch.no_grad():
            score = model(input_tensor).squeeze().item()
            
        base_filename = os.path.basename(media_data['filepath']).replace('.npz', '')
        viz_path = os.path.join(CONFIG['gradcam_output_dir'], f"face_quality_{base_filename}.png")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 11))
        fig.suptitle(f"Face Quality Analysis (Model Score: {score:.3f})", fontsize=16, weight='bold')

        axes[0, 0].imshow(face_rgb)
        axes[0, 0].set_title("Original Face (Frame 0)")
        axes[0, 0].axis('off')

        # facial Landmarks
        landmark_face_plot = None
        landmarks_found = False
        if dlib_detector is not None and dlib_predictor is not None:
            for frame in faces:
                dets = dlib_detector(frame, 1)
                if len(dets) > 0:
                    shape = dlib_predictor(frame, dets[0])
                    landmark_face_plot = frame.copy()
                    for i in range(0, shape.num_parts):
                        p = shape.part(i)
                        cv2.circle(landmark_face_plot, (p.x, p.y), 2, (0, 255, 0), -1)
                    landmarks_found = True
                    break

        if landmarks_found:
            axes[0, 1].imshow(cv2.cvtColor(landmark_face_plot, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title("Facial Landmarks (First Valid Frame)")
        else:
            axes[0, 1].imshow(face_rgb)
            axes[0, 1].set_title("Facial Landmarks (Not Detected)")
        axes[0, 1].axis('off')

        # combined exposure & blur
        combined_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        laplacian_img = cv2.Laplacian(gray, cv2.CV_64F)
        threshold = np.percentile(np.abs(laplacian_img), 99.5)
        edge_mask = np.abs(laplacian_img) > threshold
        combined_display[edge_mask] = [255, 255, 0] # Cyan in BGR
        
        axes[1, 0].imshow(cv2.cvtColor(combined_display, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f"Exposure & Blur Overlay\nExposure: {exposure_model_input:.3f} | Blur: {blur_model_input:.3f}")
        axes[1, 0].axis('off')
        
        edge_legend = mpatches.Patch(color='cyan', label='Strong Edges (Blur)')
        axes[1, 0].legend(handles=[edge_legend], loc='lower right', fontsize='small')

        # bar chart plot
        metrics = {'Blur': blur_model_input, 'Exposure': exposure_model_input}
        axes[1, 1].bar(metrics.keys(), metrics.values(), color=['#1f77b4', '#ff7f0e'])
        axes[1, 1].set_title("Summary of Model Inputs")
        axes[1, 1].set_ylabel("Normalized Value")
        axes[1, 1].set_ylim(0, max(1.0, blur_model_input * 1.2))
        axes[1, 1].axhline(y=0.1, color='r', linestyle='--', linewidth=1)
        axes[1, 1].axhline(y=0.3, color='orange', linestyle='--', linewidth=1)
        axes[1, 1].axhline(y=0.7, color='orange', linestyle='--', linewidth=1)
        axes[1, 1].text(1.02, 0.1, 'Blurry', va='center', ha='left', backgroundcolor='w')
        axes[1, 1].text(1.02, 0.5, 'Good Exposure', va='center', ha='left', backgroundcolor='w')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(viz_path, dpi=150)
        plt.close(fig)

    except Exception as e:
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        return create_error_result(agent_name, str(e))
        
    return {
        'agent': agent_name,
        'score': score,
        'details': 'Face quality analysis and visualization complete.',
        'visualization_path': viz_path
    }


# dlib setup
try:
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor(predictor_path)
except Exception as e:
    dlib_detector, dlib_predictor = None, None
    logging.error(f"Failed to initialize dlib: {e}. Landmark plotting disabled.")


def run_visual_context_analysis(media_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses a Vision-Language Model (Gemini) to identify the person and describe the visual context.
    This provides high-level semantic clues for the final report generation.
    """
    agent_name = "Visual Context"
    if not GOOGLE_API_KEY:
        return {'agent': agent_name, 'description': 'Agent skipped: Google API Key not configured.', 'error': True}

    if media_data.get('processed_faces') is None or media_data['processed_faces'].size == 0:
        return {'agent': agent_name, 'description': 'No face frames available for context analysis.', 'error': True}

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        image_np = media_data['processed_faces'][0]
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
        image_pil = Image.fromarray(image_rgb)
        
        prompt = """Analyze the person in this image. 
1. If they are a recognizable public figure, identify them by name.
2. Briefly describe the visual context (e.g., formal interview, casual setting, outdoor speech).
Respond concisely in a single paragraph."""

        response = model.generate_content([prompt, image_pil])
        description = response.text.strip()
        
        return {
            'agent': agent_name,
            'description': description,
            'details': f"Gemini-Pro Vision analysis complete.",
            'error': False
        }
    except Exception as e:
        logging.error(f"Visual Context agent failed: {e}", exc_info=True)
        return {'agent': agent_name, 'description': f"Gemini API error: {e}", 'error': True}


def run_transcription(media_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transcribes the audio waveform using a local Whisper model.
    """
    agent_name = "Audio Transcription"
    device = CONFIG['device']
    waveform = media_data.get('audio_waveform')
    
    if waveform is None or media_data.get('audio_sample_rate') is None or waveform.size < 2048:
        return {'agent': agent_name, 'transcript': 'Audio missing or too short.', 'error': True}

    try:
        local_model_path = "./whisper-tiny-local"
        processor = AutoProcessor.from_pretrained(local_model_path)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(local_model_path).to(device)

        inputs = processor(waveform, sampling_rate=media_data['audio_sample_rate'], return_tensors="pt").to(device)
        predicted_ids = model.generate(inputs.input_features)
        transcript_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        logging.info(f"Whisper Transcript: '{transcript_text}'")

        if not transcript_text.strip():
            return {'agent': agent_name, 'transcript': '', 'details': 'Audio was silent or contained no discernible speech.', 'error': False}

        return {
            'agent': agent_name,
            'transcript': transcript_text,
            'details': 'Transcription successful.',
            'error': False
        }
    except Exception as e:
        logging.error(f"{agent_name} failed: {e}", exc_info=True)
        return {'agent': agent_name, 'transcript': f"Transcription error: {e}", 'error': True}

def _prepare_report_context(decision: Dict[str, Any]) -> Dict[str, Any]:
    """Gathers all data from the decision object for report generation."""
    media_data = decision.get('media_data', {})
    filename = os.path.basename(media_data.get('filepath', 'unknown.npz'))
    
    # ground truth
    ground_truth = "Unknown"
    if 'metadata' in media_data and 'ground_truth' in media_data['metadata']:
        ground_truth = media_data['metadata']['ground_truth']

    agent_summary_lines = []
    for r in decision.get('results', []):
        agent_name = r.get('agent', 'Unknown Agent')
        # first wave
        if agent_name in ["Visual (Spatial)", "Audio (FreqNet)", "Audio Forensics (ECAPA)", "Cross-Modal (Lip-Sync)", "Face Quality"]:
            if not r.get('error'):
                agent_summary_lines.append(f"- {agent_name}: Score={r['score']:.4f}")
            else:
                agent_summary_lines.append(f"- {agent_name}: ERROR ({r.get('details', 'N/A')})")
    
    agents_used = decision.get('agents_used', [])
    first_wave = [a for a in agents_used if a in CONFIG['multi_agent']['first_wave_agents']]
    second_wave = [a for a in agents_used if a in CONFIG['multi_agent']['second_wave_agents']]
    
    deployment_info = f"- First Wave Agents Deployed: {', '.join(first_wave)}\n"
    if second_wave:
        deployment_info += f"- Second Wave Triggered (Disagreement > {CONFIG['multi_agent']['disagreement_threshold']}): {', '.join(second_wave)}"
    else:
        deployment_info += "- Second Wave Not Triggered"

    return {
        "filename": filename,
        "ground_truth": ground_truth,
        "verdict": decision['verdict'],
        "confidence": decision['confidence'],
        "threshold": CONFIG['decision_engine']['threshold'],
        "disagreement": decision.get('disagreement_level', 0),
        "analysis_time": decision.get('analysis_time', 0),
        "agent_summary": "\n".join(agent_summary_lines),
        "deployment_info": deployment_info,
        "visualizations_generated": len(decision.get('visualizations', [])),
    }
    

def generate_llama_report(decision: Dict[str, Any]):
    """Generates detailed forensic report with two-phase deployment information"""
    media_data = decision.get('media_data', {})
    filename = os.path.basename(media_data.get('filepath', 'unknown.npz'))
    
    # ground truth from npz
    metadata = {}
    if 'filepath' in media_data:
        try:
            data = np.load(media_data['filepath'], allow_pickle=True)
            if 'metadata' in data:
                metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else {}
            elif 'label' in data:
                label = data.get('label', ["Unknown"])[0]
                metadata = {'ground_truth': label.capitalize()}
        except:
            pass
    
    ground_truth = metadata.get('ground_truth', 'Unknown')
    verdict = decision['verdict']
    confidence = decision['confidence']
    results = decision['results']
    threshold = CONFIG['decision_engine']['threshold']
    disagreement = decision.get('disagreement_level', 0)

    # agent analysis summary
    agent_summary = ""
    core_agents = ["Visual (Spatial)", "Audio (FreqNet)", "Audio Forensics (ECAPA)", 
                   "Cross-Modal (Lip-Sync)", "Face Quality"]
    
    for agent_name in core_agents:
        result = next((r for r in results if r['agent'] == agent_name), None)
        if result:
            if not result.get('error'):
                score = result['score']
                status = f"Score: {score:.4f}"
            else:
                status = f"ERROR ({result.get('details', 'Unknown error')})"
            agent_summary += f"- **{agent_name}**: {status}\n"

    # semantic context
    visual_context_result = next((r for r in results if r['agent'] == 'Visual Context'), None)
    transcription_result = next((r for r in results if r['agent'] == 'Audio Transcription'), None)
    
    visual_description = "Not performed"
    if visual_context_result:
        visual_description = visual_context_result.get('description', 'Analysis failed')
    
    audio_transcript = "Not performed"
    if transcription_result:
        audio_transcript = transcription_result.get('transcript', 'Transcription failed')

    # Determine which agents were deployed
    agents_used = decision.get('agents_used', [])
    first_wave = [a for a in agents_used if a in CONFIG['multi_agent']['first_wave_agents']]
    second_wave = [a for a in agents_used if a in CONFIG['multi_agent']['second_wave_agents']]
    
    deployment_info = f"First wave: {', '.join(first_wave)}"
    if second_wave:
        deployment_info += f"\nSecond wave triggered (disagreement: {disagreement:.3f}): {', '.join(second_wave)}"
    else:
        deployment_info += f"\nSecond wave not triggered (disagreement: {disagreement:.3f} < 0.3)"

    prompt = f"""
**Role:** You are a Senior Digital Forensics Expert specializing in deepfake detection using multi-agent AI systems.

**Task:** Write a comprehensive forensic report documenting the two-phase multi-agent analysis of a media file.

---
### CASE DETAILS ###
- **File:** `{filename}`
- **Ground Truth:** `{ground_truth}`
- **Disagreement Level:** `{disagreement:.3f}` (Threshold: 0.3)
- **Processing Time:** `{decision.get('analysis_time', 0):.2f} seconds`

### TWO-PHASE DEPLOYMENT ###
{deployment_info}

### MULTI-AGENT ANALYSIS ###
**Final Verdict:** `{verdict}`
**Confidence Score:** `{confidence:.4f}` (Threshold: {threshold})

**Agent Scores (Equal Weighting):**
{agent_summary}

**Semantic Context:**
- Visual: "{visual_description}"
- Audio: "{audio_transcript}"

### XAI Visualizations Generated ###
{len(decision.get('visualizations', []))} visualization files created for explainability

---
### REPORT REQUIREMENTS ###

**1. Executive Summary:**
   - State the **System's Final Verdict** and its **Confidence Score** exactly as provided above.
   - Write a one-sentence justification that explains how the low-level scores led to this verdict (e.g., "The verdict is Real because the weighted average score of {confidence:.4f} is below the {threshold} deepfake threshold.")
   - Explain the two-phase approach and whether second wave was deployed
   - Note the disagreement level that triggered (or didn't trigger) phase 2

**2. Phase Analysis**
   - Briefly list the scores from the contributing agents
   - State plainly whether these scores are high (indicating artifacts) or low (indicating authenticity), and how this supports the final verdict.


**3. Contextual Plausibility Analysis (For Supplementary Insight Only):**
   - **Identity vs. Content Mismatch:** Based on the identified person and the transcript, does the content seem plausible or out of character? Note this as an observation.
   - **Language Mismatch:** Does the identified person speak the language in the transcript? Note this as an observation.
   - **Overall Scenario Plausibility:** Briefly comment on whether the overall scenario makes sense.

**4. Final Verification:**
   - Reiterate the **System's Final Verdict**.
   - Compare this verdict with the provided **Ground Truth Label** and state if the system was **Correct** or **Incorrect**.

"""

    try:
        if not GROQ_API_KEY or "YOUR_GROQ_API" in GROQ_API_KEY:
            raise ValueError("Groq API Key not configured")
        
        client = Groq(api_key=GROQ_API_KEY)
        logging.info(f"Generating forensic report with two-phase analysis details...")
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=1500
        )
        
        print("\n" + "="*80)
        print(f"TWO-PHASE MULTI-AGENT FORENSIC ANALYSIS REPORT")
        print(f" File: {filename.upper()}")
        print("="*80)
        print(chat_completion.choices[0].message.content)
        print("="*80)
        
        # Print deployment visualization
        print("\nDEPLOYMENT PHASES:")
        print("-" * 40)
        print(f"Phase 1 - First Wave: {', '.join(first_wave)}")
        if second_wave:
            print(f"Phase 2 - Second Wave: {', '.join(second_wave)}")
            print(f"Reason: Disagreement level {disagreement:.3f} > 0.3")
        else:
            print(f"Phase 2 - Not triggered")
            print(f"Reason: Disagreement level {disagreement:.3f} < 0.3")
        print("-" * 40)
        
    except Exception as e:
        logging.warning(f"LLM report generation failed: {e}. Using enhanced fallback.")
        
        # Enhanced fallback report
        print("\n" + "="*80)
        print(f"ANALYSIS REPORT: {filename.upper()}")
        print("="*80)
        print(f"\nVERDICT: {verdict}")
        print(f"CONFIDENCE: {confidence:.2%}")
        print(f"THRESHOLD: {threshold}")
        print(f"GROUND TRUTH: {ground_truth}")
        print(f"ASSESSMENT: {'CORRECT' if verdict == ground_truth.upper() else 'INCORRECT'}")
        print(f"\nDISAGREEMENT LEVEL: {disagreement:.3f}")
        print(f"PROCESSING TIME: {decision.get('analysis_time', 0):.2f}s")
        
        print("\n--- Two-Phase Deployment ---")
        print(f"First Wave: {', '.join(first_wave)}")
        if second_wave:
            print(f"Second Wave (triggered): {', '.join(second_wave)}")
        else:
            print(f"Second Wave: Not triggered (disagreement < 0.3)")
        
        print("\n--- Agent Analysis ---")
        for r in results:
            if not r.get('error'):
                agent = r['agent']
                score = r.get('score', -1)
                if score >= 0:
                    print(f"  ✓ {agent:<25}: Score={score:.3f}")
            else:
                print(f"{r['agent']:<25}: ERROR")
        
        if decision.get('visualizations'):
            print(f"\n--- XAI Visualizations ---")
            for viz in decision['visualizations']:
                print(f"{viz['agent']}: {viz['path']}")
        
        print("="*80 + "\n")
        

def main():

    parser = argparse.ArgumentParser(description="Multi-Agent Deepfake Detection Analysis")
    parser.add_argument(
        '--file', 
        type=str, 
        default=None,
        help='Specify a unique part of the filename (e.g., "EN_1234_FAKE") to process.'
    )
    args = parser.parse_args()

    setup_debug_logging()
    
    models = load_all_models()
    if not models:
        logging.critical("No models loaded successfully. Exiting.")
        return
    
    orchestrator = MultiAgentOrchestrator(models)

    test_files = debug_file_discovery()
    if not test_files:
        logging.error("No NPZ files found for processing. Exiting.")
        return
        

    selected_file = None
    if args.file:
        target_substring = os.path.basename(args.file).replace('.npz', '')

        found_files = [f for f in test_files if target_substring in f]
        
        if len(found_files) == 1:
            selected_file = found_files[0]
            logging.info(f"Target file specified and uniquely found: {os.path.basename(selected_file)}")
        elif len(found_files) > 1:
            logging.error(f"Ambiguous file name! Found {len(found_files)} matches for '{target_substring}'. Be more specific.")
            logging.info(f"Matches found: {[os.path.basename(f) for f in found_files]}")
            return
        else:
            logging.error(f"Specified file containing '{target_substring}' not found in the test set. Exiting.")
            return
    else:
        # No file specified, fall back to the old behavior
        logging.warning("No target file specified. Processing the first file found by default.")
        selected_file = test_files[0]

    if selected_file is None:
        logging.error("No file was selected for processing. Exiting.")
        return
    
    logging.info(f"Processing single file: {os.path.basename(selected_file)}")
    try:
        media_data_dict = debug_file_loading(selected_file)
        if media_data_dict:
            # core agents
            final_decision = orchestrator.orchestrate_analysis(media_data_dict)
            final_decision['media_data'] = media_data_dict
            
            # semantic context
            logging.info("Gathering semantic context for report...")
            visual_context = run_visual_context_analysis(media_data_dict, models)
            transcription = run_transcription(media_data_dict, models)
            
            # add semantic results
            final_decision.setdefault('results', []).extend([visual_context, transcription])

            # enhanced, context-aware LLM report
            generate_llama_report(final_decision)
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
    
    logging.info(f"Check output directory: {CONFIG['gradcam_output_dir']}")
    
if __name__ == "__main__":
    main()

