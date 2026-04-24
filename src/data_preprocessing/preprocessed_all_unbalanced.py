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

DATA_DIR = 'polyglot_lang'


warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def process_and_save_files(file_list, output_dir, detector, args):
    """
    Processes a list of video files, saves the data, and returns a list of any files
    that were found to have no audio.
    """
    if not file_list:
        return []
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nProcessing {len(file_list)} files for the '{os.path.basename(output_dir)}' set...")

    files_with_no_audio = []

    for video_path, label in tqdm(file_list, desc=f"Processing {os.path.basename(output_dir)}"):
        print(f"Processing video: {video_path} with label: {label}")
        video_basename = os.path.basename(video_path).replace('.mp4', '')
        output_filename = os.path.join(output_dir, f"{video_basename}_label_{'fake' if int(label) == 1 else 'real'}.npz")
        if os.path.exists(output_filename):
            continue


        faces = []
        try:
            cap = cv2.VideoCapture(video_path)
            

            frame_count = 0
            
            while cap.isOpened() and len(faces) < args.max_faces:
                ret, frame = cap.read()
                if not ret: break
                
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
                                resized_face = cv2.resize(face_crop, (args.image_size, args.image_size))
                                faces.append(resized_face)
                                if len(faces) >= args.max_faces: break
                
                frame_count += 1
            cap.release()
        except Exception as e:
            tqdm.write(f"Warning: Could not process video frames from {video_basename}. Error: {e}")
            continue


        if not faces:
            tqdm.write(f"Warning: No faces found in {video_basename}. Skipping file save.")
            continue

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
            tqdm.write(f"-> Logging {video_basename} for audio issue report.")
            files_with_no_audio.append(video_path)

        try:
            string_label = 'fake' if int(label) == 1 else 'real'
            
            tqdm.write(f"-> SAVING: '{output_filename}' with internal label: '{string_label}'")
            

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
    """Main function to orchestrate the data processing and splitting pipeline."""

    LANGUAGES = ['ar', 'en', 'es', 'fr', 'ja', 'ru', 'zh']
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)


    combined_train_files = []
    combined_val_files = []
    combined_test_files = []

    for lang in LANGUAGES:
        print(f"\n---> Processing and splitting language: {lang.upper()}")
        lang_real_files = []
        lang_fake_files = []

        # Load REAL videos
        real_json_path = os.path.join(DATA_DIR, 'json_file', 'real_json_file', f'{lang}.json')
        if os.path.exists(real_json_path):
            with open(real_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            video_entries = data.get('video') or data.get('videos', []) if isinstance(data, dict) else data
            for entry in video_entries:
                filename = entry.get('filename') or entry.get('name')
                if not filename: continue
                full_path = os.path.join(DATA_DIR, 'real', lang, filename + ('' if filename.endswith('.mp4') else '.mp4'))
                if os.path.exists(full_path):
                    lang_real_files.append((full_path, 0))

        # Load fake videos
        fake_json_path = os.path.join(DATA_DIR, 'json_file', 'fake_Json_file', f'to_{lang}.json')
        if os.path.exists(fake_json_path):
            with open(fake_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            video_entries = data.get('video') or data.get('videos', []) if isinstance(data, dict) else data
            for entry in video_entries:
                filename = entry.get('filename') or entry.get('name')
                if not filename: continue
                full_path = os.path.join(DATA_DIR, 'fake', f'to_{lang}', filename + ('' if filename.endswith('.mp4') else '.mp4'))
                if os.path.exists(full_path):
                    lang_fake_files.append((full_path, 1))
        
        lang_real_files.sort()
        lang_fake_files.sort()
        

        # Split data
        if lang_real_files:
            real_train_val, real_test = train_test_split(lang_real_files, test_size=0.15, random_state=RANDOM_SEED)
            real_train, real_val = train_test_split(real_train_val, test_size=0.1765, random_state=RANDOM_SEED)
            combined_train_files.extend(real_train)
            combined_val_files.extend(real_val)
            combined_test_files.extend(real_test)

        if lang_fake_files:
            fake_train_val, fake_test = train_test_split(lang_fake_files, test_size=0.15, random_state=RANDOM_SEED)
            fake_train, fake_val = train_test_split(fake_train_val, test_size=0.1765, random_state=RANDOM_SEED)
            combined_train_files.extend(fake_train)
            combined_val_files.extend(fake_val)
            combined_test_files.extend(fake_test)

 
    np.random.shuffle(combined_train_files)
    np.random.shuffle(combined_val_files)
    np.random.shuffle(combined_test_files)
    

    # Balance Train Set by downsampling the majority class
    train_reals = [f for f in combined_train_files if f[1] == 0]
    train_fakes = [f for f in combined_train_files if f[1] == 1]
    min_train = min(len(train_reals), len(train_fakes))
    train_list = train_reals[:min_train] + train_fakes[:min_train]
    np.random.shuffle(train_list)
    print(f"  - Train set balanced: {len(train_list)} total files ({min_train} real, {min_train} fake).")

    # Balance Validation Set by downsampling the majority class
    val_reals = [f for f in combined_val_files if f[1] == 0]
    val_fakes = [f for f in combined_val_files if f[1] == 1]
    min_val = min(len(val_reals), len(val_fakes))
    val_list = val_reals[:min_val] + val_fakes[:min_val]
    np.random.shuffle(val_list)
    print(f"  - Validation set balanced: {len(val_list)} total files ({min_val} real, {min_val} fake).")


    test_list = combined_test_files
    test_reals_count = len([f for f in test_list if f[1] == 0])
    test_fakes_count = len([f for f in test_list if f[1] == 1])
    print(f"  - Test set prepared (unbalanced): {len(test_list)} total files ({test_reals_count} real, {test_fakes_count} fake).")


    detector = MTCNN()
    
    # Process all three sets
    process_and_save_files(train_list, os.path.join(args.output_dir, 'train'), detector, args)
    process_and_save_files(val_list, os.path.join(args.output_dir, 'val'), detector, args)
    process_and_save_files(test_list, os.path.join(args.output_dir, 'test'), detector, args)

    print("\n--- All Steps Complete ---")
    print(f"Processed data is located in: {args.output_dir}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-process and split Polyglot data using JSON manifests.")
    
    parser.add_argument('--output_dir', type=str, default='polyglot_processed_all_unbalanced', 
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