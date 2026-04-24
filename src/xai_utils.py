
"""
A collection of XAI utilities for multi-modal deepfake detection.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import tensorflow as tf  # only needed by legacy Keras-model helpers below
except ImportError:  # pragma: no cover
    tf = None
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import librosa
import librosa.display
import logging
from typing import Optional, Tuple, Dict, Any, Union

# Make sure matplotlib works on servers without a display.
import matplotlib
matplotlib.use('Agg')


class GradCAM:
    """A robust Grad-CAM implementation for PyTorch that handles model evaluation state."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hooks = []
        self._register_forward_hook()

    def _forward_hook(self, module, input, output):
        # Hook to store the activations from the target layer
        self.activations = output

    def _backward_hook(self, grad):
        # Hook to store the gradients from the backward pass
        self.gradients = grad

    def _register_forward_hook(self):
        # Sets up the forward hook on the target layer
        self.remove_hooks()
        handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.hooks.append(handle)

    def remove_hooks(self):
        # Clean up any existing hooks
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def __call__(self, input_tensor, target_category=None):
        # Save original grad states and enable gradients for all params.
        # This is important if the model was in torch.no_grad() mode.
        original_grad_states = {}
        for name, param in self.model.named_parameters():
            original_grad_states[name] = param.requires_grad
            param.requires_grad_(True)
        
        try:
            self.model.eval()
            self.model.zero_grad()
            
            # --- 1. Forward pass ---
            # We need to build the graph, so we run this in an enable_grad context.
            with torch.enable_grad():
                output = self.model(input_tensor)
            
            if self.activations is None:
                raise ValueError("Grad-CAM Error: Activations were not captured.")
            
            # --- 2. Register backward hook on the activations tensor itself ---
            activations_hook_handle = self.activations.register_hook(self._backward_hook)
            
            # --- 3. Backward pass ---
            score = output.squeeze() if target_category is None else output[:, target_category]
            score.backward(torch.ones_like(score))
            
            # --- 4. Check for gradients and clean up the hook ---
            activations_hook_handle.remove()
            if self.gradients is None:
                raise ValueError("Grad-CAM Error: Gradients were not captured.")

            # --- 5. Compute the heatmap ---
            pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
            # Weight the channels by the gradients
            for i in range(self.activations.shape[1]):
                self.activations[:, i, :, :] *= pooled_grads[i]
            
            heatmap = torch.mean(self.activations, dim=1).squeeze().cpu()
            heatmap = F.relu(heatmap)
            
            # Normalize to [0, 1]
            if torch.max(heatmap) > 0:
                heatmap /= torch.max(heatmap)
            
            return heatmap.detach().numpy()

        finally:
            # --- ALWAYS restore original gradient states ---
            for name, param in self.model.named_parameters():
                param.requires_grad_(original_grad_states[name])
            
            self.gradients = None
            self.activations = None

def generate_keras_gradcam(model, img_array, layer_name="block14_sepconv2_act"):
    """Creates a Grad-CAM heatmap for a Keras model."""
    try:
        # Create a sub-model that outputs the feature map and the final prediction
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            class_channel = preds[:, 0]
        
        # Get the gradients and pool them
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature map and create the heatmap
        heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        
        return heatmap.numpy()
    
    except Exception as e:
        logging.error(f"Keras Grad-CAM generation failed: {e}")
        return np.zeros((1, 1))


def create_visual_overlay(original_frame_rgb, heatmap):
    """Overlays a heatmap onto an RGB image."""
    # Make sure the heatmap is a valid numpy array
    if heatmap is None or not np.all(np.isfinite(heatmap)):
        logging.warning("Got a bad heatmap (None or NaN). Returning the original frame.")
        return original_frame_rgb

    # Heatmap must be a 2D image.
    if heatmap.ndim != 2:
        logging.warning(f"Heatmap isn't a 2D image (shape: {heatmap.shape}). Can't create overlay.")
        return original_frame_rgb
        
    try:
        # Resize heatmap to match the frame and apply a colormap
        heatmap_resized = cv2.resize(heatmap, (original_frame_rgb.shape[1], original_frame_rgb.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # OpenCV works in BGR, so we need to convert before blending
        original_frame_bgr = cv2.cvtColor(original_frame_rgb, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.addWeighted(heatmap_color, 0.4, original_frame_bgr, 0.6, 0)
        
        # Convert back to RGB for displaying with matplotlib, etc.
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        return overlay_rgb

    except cv2.error as e:
        logging.error(f"OpenCV blew up during overlay creation: {e}. Returning original frame.")
        return original_frame_rgb

def create_audio_overlay(log_mel_spec, heatmap, alpha=0.4):
    """Creates a visual overlay for a spectrogram using a heatmap."""
    try:
        # Normalize spectrogram to a 0-255 range for visualization
        spec_min, spec_max = np.min(log_mel_spec), np.max(log_mel_spec)
        spec_norm = np.uint8(255 * (log_mel_spec - spec_min) / (spec_max - spec_min)) if spec_max > spec_min else np.zeros_like(log_mel_spec, dtype=np.uint8)
        
        spec_img_bgr = cv2.applyColorMap(spec_norm, cv2.COLORMAP_MAGMA)
        
        # If the heatmap is bad, just return the base spectrogram
        if heatmap is None or heatmap.size <= 1 or not np.all(np.isfinite(heatmap)):
            logging.warning("Invalid heatmap for audio; returning spectrogram only.")
            return spec_img_bgr
        
        # Resize heatmap and apply a different colormap
        heatmap_resized = cv2.resize(heatmap, (spec_img_bgr.shape[1], spec_img_bgr.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        # Blend them together
        overlay_img = cv2.addWeighted(heatmap_color, alpha, spec_img_bgr, 1 - alpha, 0)
        return overlay_img
        
    except Exception as e:
        logging.error(f"Failed to create audio overlay: {e}")
        # Return a black image on failure
        return np.zeros((log_mel_spec.shape[0], log_mel_spec.shape[1], 3), dtype=np.uint8)

def save_audio_heatmap(log_mel_spec, heatmap, output_path):
    """Saves an audio Grad-CAM visualization to a file."""
    try:
        overlay_img = create_audio_overlay(log_mel_spec, heatmap)
        cv2.imwrite(output_path, overlay_img)
        logging.info(f"Audio heatmap saved to {output_path}")
    except Exception as e:
        logging.error(f"Couldn't save audio heatmap: {e}")

def generate_cross_attention_viz(cross_attention_weights, faces, base_filename, output_dir, fps, frame_stride):
    """Creates a filmstrip visualization of cross-attention weights."""
    viz_path = os.path.join(output_dir, f"cross_attention_{base_filename}.png")
    
    num_frames_to_plot = 10
    if not num_frames_to_plot:
        return None

    fig, axs = plt.subplots(1, num_frames_to_plot, figsize=(num_frames_to_plot * 2.5, 3))
    if num_frames_to_plot == 1:
        axs = [axs]

    try:
        weights = cross_attention_weights.squeeze().cpu().numpy()
        # Normalize weights for visualization purposes (e.g., border thickness)
        min_w, max_w = np.min(weights), np.max(weights)
        norm_weights = (weights - min_w) / (max_w - min_w + 1e-6) if max_w - min_w > 1e-6 else weights

        for i in range(num_frames_to_plot):
            try:
                frame = faces[i]
                if not isinstance(frame, np.ndarray) or frame.size < 100:
                    raise ValueError(f"Frame #{i} is invalid.")

                # Ensure frame is a 3-channel, 8-bit image
                if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame = frame.astype(np.uint8)
                
                # Use attention weight to set border thickness
                vis_weight = norm_weights[i]
                border_thickness = max(1, int(vis_weight * 15))
                frame_with_border = cv2.rectangle(frame.copy(), (0,0), (frame.shape[1]-1, frame.shape[0]-1), (0, 255, 0), border_thickness)
                frame_rgb = cv2.cvtColor(frame_with_border, cv2.COLOR_BGR2RGB)
                
                # Plot the frame and its raw weight
                axs[i].imshow(frame_rgb)
                axs[i].text(5, 15, f'{weights[i]:.3f}', color='white', weight='bold', bbox=dict(facecolor='black', alpha=0.6, pad=2))
                
                # Add a timestamp
                time_sec = (i * frame_stride) / fps
                axs[i].set_xlabel(f'{time_sec:.2f}s')
                axs[i].set_xticks([])
                axs[i].set_yticks([])

            except Exception as frame_error:
                # If one frame fails, draw an error box but don't crash the whole visualization
                logging.error(f"Couldn't plot frame #{i} for '{base_filename}': {frame_error}")
                error_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                cv2.putText(error_frame, "ERROR", (65, 112), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                axs[i].imshow(error_frame)
                axs[i].axis('off')

        fig.suptitle("Cross-Modal Attention (Audio Query -> Visual Focus)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(viz_path, dpi=150)
        
    finally:
        # Clean up the plot to free memory
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close('all')
            
    return viz_path

def generate_audio_forensics_viz(features_dict, base_filename, output_dir):
    """Creates a dashboard of audio forensic features."""
    viz_path = os.path.join(output_dir, f"audio_forensics_{base_filename}.png")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        if 'prosody' in features_dict:
            axes[0, 0].bar(range(len(features_dict['prosody'])), features_dict['prosody'], color='teal')
            axes[0, 0].set_title('Prosody Features')
        
        if 'artifacts' in features_dict:
            axes[0, 1].bar(range(len(features_dict['artifacts'])), features_dict['artifacts'], color='coral')
            axes[0, 1].set_title('Artifact Features')
        
        if 'temporal' in features_dict:
            axes[1, 0].plot(features_dict['temporal'], 'o-', color='purple')
            axes[1, 0].set_title('Temporal Consistency')
        
        if 'score' in features_dict:
            score = features_dict['score']
            axes[1, 1].barh(['Real', 'Fake'], [1-score, score], color=['green', 'red'])
            axes[1, 1].set_xlim([0, 1])
            axes[1, 1].set_title(f'Decision (Score: {score:.3f})')
        
        plt.suptitle('Audio Forensics Analysis', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        
    except Exception as e:
        logging.error(f"Audio forensics visualization failed: {e}")
        viz_path = None
        
    finally:
        plt.close('all')
    
    return viz_path

def generate_face_quality_viz(quality_metrics, face_img, base_filename, output_dir):
    """Creates a dashboard of face quality metrics."""
    viz_path = os.path.join(output_dir, f"face_quality_{base_filename}.png")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Display the face
        axes[0, 0].imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Input Face')
        axes[0, 0].axis('off')
        
        # Bar chart of quality scores
        if quality_metrics:
            metrics, values = list(quality_metrics.keys()), list(quality_metrics.values())
            axes[0, 1].barh(metrics, values, color=plt.cm.viridis(np.linspace(0, 1, len(metrics))))
            axes[0, 1].set_title('Quality Metrics')
            axes[0, 1].set_xlim([0, 1])
        
        # Pixel intensity histogram
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        axes[1, 0].hist(gray.ravel(), bins=256, color='gray')
        axes[1, 0].set_title('Intensity Distribution')
        axes[1, 0].set_yticklabels([])

        # Remove the last subplot and put text there instead
        axes[1, 1].axis('off')
        text_summary = "Face Quality Summary:\n\n" + "\n".join([f"- {k.capitalize()}: {v:.2f}" for k, v in quality_metrics.items()])
        axes[1, 1].text(0.05, 0.95, text_summary, ha='left', va='top', fontsize=10, wrap=True)

        plt.suptitle('Face Quality Analysis', fontsize=16, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        
    except Exception as e:
        logging.error(f"Face quality visualization failed: {e}")
        viz_path = None
        
    finally:
        plt.close('all')
    
    return viz_path

def find_target_layer(model, layer_type=nn.Conv2d, reverse=True):
    """Finds the last convolutional layer in a model to use with Grad-CAM."""
    # Search backwards by default to find the last conv layer
    modules = reversed(list(model.named_modules())) if reverse else model.named_modules()
    
    for name, module in modules:
        if isinstance(module, layer_type):
            logging.info(f"Found target layer for Grad-CAM: {name}")
            return module
    
    logging.warning(f"Couldn't find any {layer_type} layer in the model.")
    return None

def cleanup_matplotlib():
    """Forcefully cleans up matplotlib figures to prevent memory leaks."""
    plt.close('all')
    plt.clf()
    plt.cla()
    import gc
    gc.collect()


# Make these functions importable with 'from xai_utils import *'
__all__ = [
    'GradCAM',
    'generate_keras_gradcam',
    'create_visual_overlay',
    'create_audio_overlay',
    'save_audio_heatmap',
    'generate_cross_attention_viz',
    'generate_audio_forensics_viz',
    'generate_face_quality_viz',
    'find_target_layer',
    'cleanup_matplotlib'
]