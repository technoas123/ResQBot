"""
Audio Feature Extraction and Embedding Pipeline for Disaster Human Detection

This module processes audio files from a disaster detection dataset and extracts 
comprehensive acoustic features for classification. It performs the following:

1. Audio preprocessing: loads, trims silence, and normalizes audio signals
2. Feature extraction: computes MFCCs, spectral features (centroid, rolloff), 
   zero-crossing rate, chroma features, and RMS energy with their deltas
3. Statistical aggregation: calculates mean, standard deviation, and max values 
   across time for each feature
4. Dataset processing: processes train/val splits with 3 classes (human, emergency, 
   background) and saves features as JSON files

The extracted 128-dimensional feature vectors are designed to capture acoustic 
characteristics that distinguish human vocalizations from emergency sounds and 
background noise in disaster scenarios.

Usage:
    python script_name.py
    
Output:
    - data/processed/audio_features/train_features.json
    - data/processed/audio_features/train_labels.json
    - data/processed/audio_features/val_features.json
    - data/processed/audio_features/val_labels.json
"""


import torch
import numpy as np
import librosa
from pathlib import Path
import json
from tqdm import tqdm

class AudioFeatureExtractor:
    def __init__(self, sr=22050, feature_dim=128):
        self.sr = sr
        self.feature_dim = feature_dim
    
    def extract_features(self, audio_path):
        """Extract enhanced audio features"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            # Remove silence
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
            if len(audio_trimmed) < 512:
                audio_trimmed = audio
            
            # Normalize
            audio_trimmed = audio_trimmed / (np.max(np.abs(audio_trimmed)) + 1e-8)
            
            # Extract multiple feature types
            mfcc = librosa.feature.mfcc(y=audio_trimmed, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Additional features for better human detection
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_trimmed, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_trimmed, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_trimmed)
            chroma = librosa.feature.chroma_stft(y=audio_trimmed, sr=sr)
            rms = librosa.feature.rms(y=audio_trimmed)
            
            # Combine features
            all_features = np.vstack([
                mfcc, mfcc_delta, mfcc_delta2,
                spectral_centroid, spectral_rolloff,
                zero_crossing_rate, chroma, rms
            ])
            
            # Statistical aggregation
            feature_means = np.mean(all_features, axis=1)
            feature_stds = np.std(all_features, axis=1)
            feature_max = np.max(all_features, axis=1)
            
            features = np.concatenate([feature_means, feature_stds, feature_max])
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            features = np.zeros(self.feature_dim)
        
        # Ensure consistent feature size
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        else:
            features = features[:self.feature_dim]
        
        return features.astype(np.float32)

class AudioEmbeddingCreator:
    def __init__(self):
        self.data_root = Path("data")
        self.processed_dir = self.data_root / "processed" / "disaster_human_detection"
        self.feature_dir = self.data_root / "processed" / "audio_features"
        self.extractor = AudioFeatureExtractor()
    
    def create_embeddings(self):
        """Create audio embeddings for all dataset files"""
        print("=== CREATING AUDIO EMBEDDINGS ===")
        
        # Create feature directory
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        
        # Process train and validation splits
        for split in ['train', 'val']:
            print(f"\nProcessing {split} split...")
            split_features = {}
            split_labels = {}
            
            for class_name in ['human', 'emergency', 'background']:
                audio_dir = self.processed_dir / split / class_name
                audio_files = list(audio_dir.glob("*.wav"))
                
                print(f"  Extracting features for {class_name}...")
                
                for audio_file in tqdm(audio_files, desc=f"  {class_name}"):
                    # Extract features
                    features = self.extractor.extract_features(audio_file)
                    
                    # Store with filename as key
                    file_id = f"{split}_{class_name}_{audio_file.stem}"
                    split_features[file_id] = features.tolist()
                    split_labels[file_id] = class_name
            
            # Save features and labels
            self.save_features(split_features, split_labels, split)
        
        print("\n=== EMBEDDING CREATION COMPLETED ===")
        self.print_feature_stats()
    
    def save_features(self, features, labels, split):
        """Save features and labels to JSON files"""
        features_file = self.feature_dir / f"{split}_features.json"
        labels_file = self.feature_dir / f"{split}_labels.json"
        
        with open(features_file, 'w') as f:
            json.dump(features, f, indent=2)
        
        with open(labels_file, 'w') as f:
            json.dump(labels, f, indent=2)
        
        print(f"  Saved {len(features)} {split} samples to {features_file}")
    
    def print_feature_stats(self):
        """Print statistics about created features"""
        print("\n=== FEATURE STATISTICS ===")
        
        for split in ['train', 'val']:
            features_file = self.feature_dir / f"{split}_features.json"
            
            if features_file.exists():
                with open(features_file, 'r') as f:
                    features = json.load(f)
                
                # Count by class
                class_counts = {}
                for file_id in features.keys():
                    class_name = file_id.split('_')[1]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                print(f"\n{split.upper()} Split:")
                for class_name, count in class_counts.items():
                    print(f"  {class_name}: {count} samples")
                print(f"  TOTAL: {len(features)} samples")
                
                # Feature dimension
                sample_feature = list(features.values())[0]
                print(f"  Feature dimension: {len(sample_feature)}")

if __name__ == "__main__":
    creator = AudioEmbeddingCreator()
    creator.create_embeddings()