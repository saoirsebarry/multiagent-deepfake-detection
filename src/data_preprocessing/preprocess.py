#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocesses a video dataset for deepfake detection.

This script reads JSON manifest files to locate real and fake videos,
splits them into balanced training, validation, and test sets, then processes
each video. It extracts face crops using MTCNN and the audio waveform using
librosa, saving the results into compressed .npz files for model training.
"""

import os
import json
import cv2
import numpy as np
import librosa
from mtcnn.mtcnn import MTCNN
import argparse
import warnings
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Configuration
# Assumes data is in a folder named 'polyglot_data'.
DATA_DIR = 'polyglot_data'

# Suppress verbose warnings for a cleaner output.
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow messages

def process_and_save_files(file_list, output_dir, detector, args):
    """
    Processes a list of video files to extract faces and audio, saving the
    results into individual .npz files.
    """
    if not file_list:
        return []
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nProcessing {len(file_list)} files for the '{os.path.basename(output_dir)}' set...")

    files_with_no_audio = []

    for video_path, label in tqdm(file_list, desc=f"Processing {os.path.basename(output_dir)}"):
        video_basename = os.path.basename(video_path).replace('.mp4', '')
        output_filename = os.path.join(output_dir, f"{video_basename}_label_{'fake' if int(label) == 1 else 'real'}.npz")
        
        # Skip files that have already been processed.
        if os.path.exists(output_filename):
            continue

        # Visual Processing: Extract face crops from video frames.
        faces = []
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            while cap.isOpened() and len(faces) < args.max_faces:
                ret, frame = cap.read()
                if not ret: break
                
                # Process every Nth frame to speed up extraction.
                if frame_count % args.frame_stride == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detections = detector.detect_faces(frame_rgb)
                    for det in detections:
                        if det['confidence'] > 0.95:
                            x1, y1, width, height = det['box']
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = x1 + width, y1 + height
                            face_crop = frame[y1:y2, x1:x2]
                            if face_crop.size > 0:
                                faces.append(cv2.resize(face_crop, (args.image_size, args.image_size)))
                                if len(faces) >= args.max_faces: break
                frame_count += 1
            cap.release()
        except Exception as e:
            tqdm.write(f"Warning: Could not process video frames from {video_basename}. Error: {e}")
            continue

        if not faces:
            tqdm.write(f"Warning: No faces found in {video_basename}. Skipping file.")
            continue

        # Audio Processing: Extract the audio waveform.
        waveform = np.array([])
        audio_found = False
        try:
            loaded_waveform, _ = librosa.load(video_path, sr=args.sample_rate, mono=True)
            if loaded_waveform is not None and loaded_waveform.size > 0:
                 waveform = loaded_waveform
                 audio_found = True
        except Exception as e:
            tqdm.write(f"Warning: Could not load audio for {video_basename}. Error: {e}. Saving with empty audio.")

        if not audio_found:
            tqdm.write(f"Note: {video_basename} has no audio track.")
            files_with_no_audio.append(video_path)

        # Save the extracted data to a compressed .npz file.
        try:
            string_label = 'fake' if int(label) == 1 else 'real'
            np.savez_compressed(
                output_filename, 
                faces=np.array(faces), 
                waveform=waveform,
                label=np.array([string_label])
            )
        except Exception as e:
            tqdm.write(f"Error saving file {output_filename}. Error: {e}")
    
    return files_with_no_audio

def main(args):
    """Orchestrates the data loading, splitting, and processing pipeline."""
    

    real_json_path = os.path.join(DATA_DIR, 'json_file','real_json_file', 'en.json')
    fake_json_path = os.path.join(DATA_DIR, 'json_file', 'fake_Json_file','to_en.json')

    if not os.path.exists(real_json_path) or not os.path.exists(fake_json_path):
        raise FileNotFoundError(f"Ensure '{real_json_path}' and '{fake_json_path}' exist.")

    try:
        # Load real video paths from the JSON manifest.
        with open(real_json_path, 'r', encoding='utf-8') as f: real_data = json.load(f)
        # Flexibly handle different JSON structures ('video' vs 'videos' keys).
        real_video_entries = real_data.get('video') or real_data.get('videos', [])
        real_files = []
        for entry in real_video_entries:
            filename = entry.get('filename') or entry.get('name')
            if filename:
                full_path = os.path.join(DATA_DIR, 'real', 'en', filename if filename.endswith('.mp4') else filename + '.mp4')
                real_files.append((full_path, 0))

        # Load fake video paths.
        with open(fake_json_path, 'r', encoding='utf-8') as f: fake_data = json.load(f)
        fake_video_entries = fake_data.get('video') or fake_data.get('videos', [])
        fake_files = []
        for entry in fake_video_entries:
            filename = entry.get('name') or entry.get('filename')
            if filename:
                full_path = os.path.join(DATA_DIR, 'fake', 'to_en', filename if filename.endswith('.mp4') else filename + '.mp4')
                fake_files.append((full_path, 1))

    except Exception as e:
        print(f"Error: Failed to parse JSON files. Details: {e}")
        return

    print(f"Loaded {len(real_files)} REAL and {len(fake_files)} FAKE video entries.")
    
    if not real_files and not fake_files:
        print("\nWarning: No video files were loaded. Check your JSON files and data directory.")
        return

    # Step 2: Create Train, Validation, and Test Splits (approx. 70/15/15)

    real_train_val, real_test = train_test_split(real_files, test_size=0.15, random_state=42)
    real_train, real_val = train_test_split(real_train_val, test_size=0.1765, random_state=42)

    fake_train_val, fake_test = train_test_split(fake_files, test_size=0.15, random_state=42)
    fake_train, fake_val = train_test_split(fake_train_val, test_size=0.1765, random_state=42)


    # Balance the training set by downsampling the larger class.
    min_train = min(len(real_train), len(fake_train))
    train_list = list(np.random.permutation(real_train)[:min_train]) + list(np.random.permutation(fake_train)[:min_train])
    np.random.shuffle(train_list)
    print(f"  - Train set balanced: {len(train_list)} total files ({min_train} real, {min_train} fake).")
    
    # Balance the validation set.
    min_val = min(len(real_val), len(fake_val))
    val_list = list(np.random.permutation(real_val)[:min_val]) + list(np.random.permutation(fake_val)[:min_val])
    np.random.shuffle(val_list)
    print(f"  - Validation set balanced: {len(val_list)} total files ({min_val} real, {min_val} fake).")

    # Balance the test set.
    min_test = min(len(real_test), len(fake_test))
    test_list = list(np.random.permutation(real_test)[:min_test]) + list(np.random.permutation(fake_test)[:min_test])
    np.random.shuffle(test_list)
    print(f"  - Test set balanced: {len(test_list)} total files ({min_test} real, {min_test} fake).")
    
 
    detector = MTCNN()
    
    process_and_save_files(train_list, os.path.join(args.output_dir, 'train'), detector, args)
    process_and_save_files(val_list, os.path.join(args.output_dir, 'val'), detector, args)
    process_and_save_files(test_list, os.path.join(args.output_dir, 'test'), detector, args)

    print("\nAll Steps Complete")
    print(f"Processed and split data is located in: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-process and split Polyglot data using JSON manifests.")
    
    parser.add_argument('--output_dir', type=str, default='polyglot_processed', 
                        help='Directory to save the processed and split .npz files.')
    parser.add_argument('--image_size', type=int, default=299, 
                        help='The size to resize all cropped faces to.')
    parser.add_argument('--max_faces', type=int, default=20, 
                        help='Maximum number of faces to extract from each video.')
    parser.add_argument('--frame_stride', type=int, default=10, 
                        help='Process every Nth frame to speed up extraction.')
    parser.add_argument('--sample_rate', type=int, default=16000, 
                        help='Sample rate for audio extraction.')

    args = parser.parse_args()
    main(args)