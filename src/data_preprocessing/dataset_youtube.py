#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube Face-Audio Extractor for Deepfake Dataset Creation

This script automates the process of building a deepfake detection dataset.
It can either download videos from specified YouTube channels or use local video
files. For each video, it extracts sequences of 10 frames containing a face,
along with the corresponding audio segment, and saves them into a structured
.npz format compatible with polyglot data loaders.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import warnings
from datetime import datetime
import cv2
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from mtcnn.mtcnn import MTCNN
import logging

# Configure logging for clean console output.
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings from underlying libraries.
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class FaceAudioExtractor:
    """Extracts sequences of face frames and corresponding audio from videos."""
    def __init__(self, min_face_size=60, min_confidence=0.90, sample_rate=16000):
        self.detector = MTCNN()
        self.min_face_size = min_face_size
        self.min_confidence = min_confidence
        self.sample_rate = sample_rate
        logger.info("Face detection system initialized.")

    def extract_sequences_to_npz(self, video_path, output_dir, label, max_sequences=5, 
                                 frame_count=10, image_size=299, frame_stride=10, 
                                 audio_duration_seconds=5):
        """
        Extracts multiple sequences of face frames and audio from a single video
        and saves each as a separate .npz file.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate the total number of frames a single sequence spans.
        frame_span = (frame_count - 1) * frame_stride + 1
        temporal_span_seconds = frame_span / fps
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        saved_files = []
        
        # Load the entire audio waveform once for efficiency.
        try:
            waveform, sr = librosa.load(video_path, sr=self.sample_rate, mono=True)
            total_audio_duration = len(waveform) / self.sample_rate
        except Exception as e:
            logger.warning(f"Could not load audio for {video_path}: {e}")
            waveform = np.array([])
            total_audio_duration = 0
        
        # Scan the video for valid sequences.
        sample_interval = frame_span + 30  # Move forward enough to avoid overlap.
        current_frame = 0
        sequences_found = 0
        
        with tqdm(total=max_sequences, desc="    Extracting", leave=False) as pbar:
            while current_frame <= total_frames - frame_span and sequences_found < max_sequences:
                sequence_data = self._get_frames_with_stride(
                    cap, current_frame, frame_count, image_size, frame_stride
                )
                
                if sequence_data:
                    # If faces are found, extract the corresponding audio segment.
                    sequence_mid_time = (current_frame + frame_span / 2) / fps
                    audio_start_time = max(0, sequence_mid_time - audio_duration_seconds / 2)
                    audio_end_time = min(total_audio_duration, audio_start_time + audio_duration_seconds)
                    
                    start_sample = int(audio_start_time * self.sample_rate)
                    end_sample = int(audio_end_time * self.sample_rate)
                    audio_segment = waveform[start_sample:end_sample]
                    
                    # Pad audio if it's shorter than the target duration.
                    target_length = int(audio_duration_seconds * self.sample_rate)
                    if len(audio_segment) < target_length:
                        audio_segment = np.pad(audio_segment, (0, target_length - len(audio_segment)))
                    
                    # Save the sequence as an NPZ file.
                    output_filename = f"{base_name}_seq{sequences_found:03d}_label_{label}.npz"
                    output_path = os.path.join(output_dir, output_filename)
                    try:
                        np.savez_compressed(output_path,
                            faces=np.array(sequence_data['faces']),
                            waveform=audio_segment,
                            label=np.array([label]),
                            metadata={'frame_indices': sequence_data['frame_indices']}
                        )
                        saved_files.append(output_path)
                        sequences_found += 1
                        pbar.update(1)
                        current_frame += sample_interval # Jump forward to find the next sequence.
                    except Exception as e:
                        logger.error(f"Error saving NPZ file: {e}")
                        current_frame += 30
                else:
                    current_frame += 30  # Step forward if no valid sequence is found.
        
        cap.release()
        logger.info(f"    Saved {sequences_found} sequences from {os.path.basename(video_path)}")
        return saved_files
    
    def _get_frames_with_stride(self, cap, start_frame, num_frames, image_size, frame_stride):
        """
        Attempts to find a sequence of frames, separated by a stride,
        where each sampled frame contains a valid face.
        """
        faces, frame_indices = [], []
        for i in range(num_frames):
            frame_idx = start_frame + (i * frame_stride)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: return None
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_faces(frame_rgb)
            if not detections: return None
            
            best_face = max(detections, key=lambda x: x['confidence'])
            if best_face['confidence'] < self.min_confidence: return None
            
            x, y, w, h = best_face['box']
            face_crop = frame[max(0, y):y+h, max(0, x):x+w]
            if face_crop.size == 0: return None
            
            faces.append(cv2.resize(face_crop, (image_size, image_size)))
            frame_indices.append(frame_idx)
        
        return {'faces': faces, 'frame_indices': frame_indices}


def download_videos_from_channel(channel_url, temp_dir, max_videos, channel_idx):
    """Downloads a specified number of videos from a YouTube channel."""
    channel_dir = os.path.join(temp_dir, f"channel_{channel_idx}")
    os.makedirs(channel_dir, exist_ok=True)
    
    command = [ 'yt-dlp',
        '--max-downloads', str(max_videos),
        '--match-filter', 'duration > 60 & duration < 600', # Filter by duration.
        '-f', 'best[height<=1080]/best',  # Download a reasonable resolution.
        '-o', os.path.join(channel_dir, '%(title).100s_%(id)s.%(ext)s'),
        '--ignore-errors', '--quiet', '--no-warnings', '--progress',
        f"{channel_url.strip('/')}/videos"
    ]
    
    try:
        subprocess.run(command, check=True)
        videos = [f for f in os.listdir(channel_dir) if f.endswith(('.mp4', '.webm', '.mkv'))]
        return len(videos)
    except Exception as e:
        logger.error(f"Download error for {channel_url}: {e}")
        return 0

def use_local_videos(category, output_dir, max_videos):
    """Copies local video files to a temporary directory for processing."""
    local_dir = f"./local_videos/{category}"
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        logger.warning(f"Directory created: {local_dir}. Please add {category} videos here.")
        return []
    
    video_files = [f for f in os.listdir(local_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        logger.warning(f"No video files found in {local_dir}")
        return []
    
    logger.info(f"Found {len(video_files)} local {category} videos.")
    processed = [shutil.copy2(os.path.join(local_dir, f), output_dir) for f in video_files[:max_videos]]
    return processed

def process_category(channels, category, args, extractor):
    """Processes all videos for a given category (real or fake)."""
    logger.info(f"\n{'='*60}\nProcessing {category.upper()} videos\n{'='*60}")
    
    category_dir = os.path.join(args.base_dir, "temp", category)
    npz_dir = os.path.join(args.base_dir, "raw_npz", category)
    os.makedirs(category_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)
    
    all_npz_files = []
    
    # Process videos from YouTube channels if provided.
    for idx, channel_url in enumerate(channels, 1):
        logger.info(f"\n[{idx}/{len(channels)}] Channel: {channel_url}")
        videos_count = download_videos_from_channel(channel_url, category_dir, args.max_videos, idx)
        if videos_count > 0:
            logger.info(f"Downloaded {videos_count} videos.")
            channel_video_dir = os.path.join(category_dir, f"channel_{idx}")
            video_files = [os.path.join(channel_video_dir, f) for f in os.listdir(channel_video_dir) if f.endswith(('.mp4', '.webm', '.mkv'))]
            for video_path in video_files:
                npz_files = extractor.extract_sequences_to_npz(
                    video_path, npz_dir, 'fake' if category == 'fake' else 'real',
                    max_sequences=args.max_sequences_per_video,
                    image_size=args.image_size
                )
                all_npz_files.extend(npz_files)
                os.remove(video_path) # Clean up video file after processing.
            shutil.rmtree(channel_video_dir)
    
    # Process local videos if enabled.
    if args.use_local:
        logger.info(f"\nProcessing local video files for {category}...")
        local_videos = use_local_videos(category, category_dir, args.max_videos * max(1, len(channels)))
        for video_path in local_videos:
            npz_files = extractor.extract_sequences_to_npz(
                video_path, npz_dir, 'fake' if category == 'fake' else 'real',
                max_sequences=args.max_sequences_per_video,
                image_size=args.image_size
            )
            all_npz_files.extend(npz_files)
            os.remove(video_path)
    
    logger.info(f"\nTotal {category} NPZ files created: {len(all_npz_files)}")
    return all_npz_files

def split_npz_files(base_dir, split_ratios=(0.7, 0.15, 0.15)):
    """Splits all generated NPZ files into balanced train/val/test sets."""
    raw_npz_dir = os.path.join(base_dir, "raw_npz")
    real_files = [os.path.join(raw_npz_dir, "real", f) for f in os.listdir(os.path.join(raw_npz_dir, "real")) if f.endswith('.npz')]
    fake_files = [os.path.join(raw_npz_dir, "fake", f) for f in os.listdir(os.path.join(raw_npz_dir, "fake")) if f.endswith('.npz')]
    
    if not real_files or not fake_files:
        logger.warning("Not enough data for both classes to create splits.")
        return
    
    logger.info(f"\n{'='*60}\nCreating train/val/test splits\n{'='*60}")
    
    # Split real and fake files separately.
    real_train_val, real_test = train_test_split(real_files, test_size=split_ratios[2], random_state=42)
    real_train, real_val = train_test_split(real_train_val, test_size=split_ratios[1]/(split_ratios[0]+split_ratios[1]), random_state=42)
    
    fake_train_val, fake_test = train_test_split(fake_files, test_size=split_ratios[2], random_state=42)
    fake_train, fake_val = train_test_split(fake_train_val, test_size=split_ratios[1]/(split_ratios[0]+split_ratios[1]), random_state=42)
    
    # Balance each split by downsampling the majority class.
    splits = {'train': (real_train, fake_train), 'val': (real_val, fake_val), 'test': (real_test, fake_test)}
    for split_name, (real_split, fake_split) in splits.items():
        split_dir = os.path.join(base_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        min_count = min(len(real_split), len(fake_split))
        
        balanced_files = random.sample(real_split, min_count) + random.sample(fake_split, min_count)
        for src_path in balanced_files:
            shutil.copy2(src_path, split_dir)
        logger.info(f"{split_name}: {min_count * 2} files ({min_count} real, {min_count} fake)")
    
    shutil.rmtree(raw_npz_dir)

def check_dependencies():
    """Checks for required command-line tools."""
    logger.info("Checking dependencies...")
    if not all(shutil.which(cmd) for cmd in ['yt-dlp', 'ffmpeg']):
        logger.error("Missing dependency: yt-dlp or ffmpeg is not in your system's PATH.")
        return False
    logger.info("All dependencies found.")
    return True

def main(args):
    """Main execution function."""
    logger.info("\nFace-Audio Sequence Extractor")
    if not check_dependencies(): return
    
    os.makedirs(args.base_dir, exist_ok=True)
    if args.use_local:
        logger.info("\nLocal mode enabled - using local video files.")
        os.makedirs("./local_videos/real", exist_ok=True)
        os.makedirs("./local_videos/fake", exist_ok=True)
    
    logger.info("\nInitializing face-audio extraction system...")
    extractor = FaceAudioExtractor(sample_rate=args.sample_rate)
    
    # Define your YouTube channels here.
    REAL_CHANNELS = ["https://www.youtube.com/@BBCNews"]
    FAKE_CHANNELS = ["https://www.youtube.com/@HeyGen_Official"]

    process_category(REAL_CHANNELS, "real", args, extractor)
    process_category(FAKE_CHANNELS, "fake", args, extractor)
    
    split_npz_files(args.base_dir)
    
    # Clean up the temporary download directory.
    temp_dir = os.path.join(args.base_dir, "temp")
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    
    logger.info(f"\n{'='*60}\nEXTRACTION COMPLETE\n{'='*60}")
    logger.info(f"Dataset saved to: {args.base_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 10-frame sequences with faces and audio, saving as NPZ files.")
    parser.add_argument("--base_dir", type=str, default="face_audio_dataset_hq", help="Output directory for the final dataset.")
    parser.add_argument("--max_videos", type=int, default=5, help="Max videos to download per YouTube channel.")
    parser.add_argument("--max_sequences_per_video", type=int, default=5, help="Max sequences to extract from each video.")
    parser.add_argument("--frame_count", type=int, default=10, help="Number of frames per sequence.")
    parser.add_argument("--image_size", type=int, default=299, help="Size to resize face crops to.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate.")
    parser.add_argument("--use_local", action="store_true", help="Use local videos instead of downloading from YouTube.")
    
    args = parser.parse_args()
    main(args)