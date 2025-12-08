"""
Disaster Audio Dataset Preparation and Curation Pipeline

This module aggregates and preprocesses audio samples from multiple public datasets
to create a balanced, task-specific dataset for training the disaster human detection
system. Combines human vocalizations, emergency sounds, and background noise for
robust multi-class audio classification.

Dataset Sources and Categories:

1. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
   → All samples mapped to 'human' class
   → Contains emotional speech and vocalizations (anger, fear, sadness, etc.)
   
2. CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
   → All samples mapped to 'human' class
   → Human speech with various emotional expressions

3. ESC-50 (Environmental Sound Classification)
   → Categorized mapping:
     • Human: crying_baby, sneezing, coughing, breathing, snoring, clapping
     • Emergency: siren, fire_alarm, chainsaw, helicopter
     • Background: rain, wind, thunderstorm, waves, crickets, frog, bird

4. UrbanSound8K
   → Categorized mapping:
     • Human: children_playing, street_music
     • Emergency: siren, car_horn
     • Background: air_conditioner, drilling, jackhammer, dog_bark, engine_idling

Processing Pipeline:
1. Collects audio files from all available datasets in data/raw/
2. Categorizes into 3 classes: human, emergency, background
3. Balances classes (target: 2000 samples per class or maximum available)
4. Splits into train (80%) and validation (20%) sets
5. Copies organized files to data/processed/disaster_human_detection/

Class Balancing Strategy:
- Target: 2000 samples per class (adjustable)
- If insufficient samples: uses all available without oversampling
- Random sampling to prevent overfitting to specific datasets
- Equal representation prevents model bias toward majority class

Output Structure:
data/processed/disaster_human_detection/
├── train/
│   ├── human/         (human vocalizations, speech, cries)
│   ├── emergency/     (sirens, alarms, urgent sounds)
│   └── background/    (ambient noise, nature, machinery)
└── val/
    ├── human/
    ├── emergency/
    └── background/

Usage:
    python script_name.py
    
Output:
    - Organized .wav files in train/val splits
    - Console statistics showing sample counts per class
    - Ready for feature extraction and model training

Use Case: Creates training data for the audio component of ResQBot's multi-modal
human detection system, enabling the model to distinguish human distress signals
from emergency equipment and environmental noise in disaster scenarios.
"""

import os
import shutil
import random
import pandas as pd
from pathlib import Path
import librosa
import numpy as np

class DisasterAudioPreparer:
    def __init__(self):
        self.data_root = Path("data")
        self.raw_dir = self.data_root / "raw"
        self.processed_dir = self.data_root / "processed" / "disaster_human_detection"
        
    def prepare_dataset(self):
        """Prepare balanced disaster human detection dataset"""
        print("=== PREPARING DISASTER AUDIO DATASET ===")
        
        # Clear existing processed data
        if self.processed_dir.exists():
            shutil.rmtree(self.processed_dir)
        
        # Create directories
        for split in ['train', 'val']:
            for class_name in ['human', 'emergency', 'background']:
                (self.processed_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Collect samples from all datasets
        all_samples = self.collect_samples()
        
        # Balance and split
        balanced_samples = self.balance_classes(all_samples)
        self.split_and_save(balanced_samples)
        
        print("\n=== DATASET PREPARATION COMPLETED ===")
        self.print_dataset_stats()
    
    def collect_samples(self):
        """Collect samples from all available datasets"""
        samples = {'human': [], 'emergency': [], 'background': []}
        
        print("\nCollecting samples from datasets...")
        
        # RAVDESS - All human vocalizations
        ravdess_path = self.raw_dir / "specialized_dataset_audio" / "ravdess"
        if ravdess_path.exists():
            print(" Processing RAVDESS...")
            samples = self.process_ravdess(ravdess_path, samples)
        
        # CREMA-D - All human speech
        crema_path = self.raw_dir / "specialized_dataset_audio" / "CREMA-D"
        if crema_path.exists():
            print(" Processing CREMA-D...")
            samples = self.process_crema_d(crema_path, samples)
        
        # ESC-50 - Categorized sounds
        esc50_path = self.raw_dir / "primary_dataset_audio" / "esc50"
        if esc50_path.exists():
            print(" Processing ESC-50...")
            samples = self.process_esc50(esc50_path, samples)
        
        # UrbanSound8K - Categorized sounds
        urban_path = self.raw_dir / "primary_dataset_audio" / "UrbanSound8K"
        if urban_path.exists():
            print(" Processing UrbanSound8K...")
            samples = self.process_urban_sound(urban_path, samples)
        
        return samples
    
    def process_ravdess(self, ravdess_path, samples):
        """Process RAVDESS dataset - all files are human vocalizations"""
        for audio_file in ravdess_path.rglob("*.wav"):
            samples['human'].append(audio_file)
        print(f"  Added {len(samples['human'])} RAVDESS samples to human class")
        return samples
    
    def process_crema_d(self, crema_path, samples):
        """Process CREMA-D dataset - all files are human speech"""
        audio_wav_path = crema_path / "AudioWAV"
        if audio_wav_path.exists():
            for audio_file in audio_wav_path.rglob("*.wav"):
                samples['human'].append(audio_file)
        print(f"  Added {len(samples['human'])} CREMA-D samples to human class")
        return samples
    
    def process_esc50(self, esc50_path, samples):
        """Process ESC-50 dataset with categorization"""
        audio_dir = esc50_path / "audio"
        meta_file = esc50_path / "meta" / "esc50.csv"
        
        if not audio_dir.exists():
            return samples
        
        # ESC-50 category mapping
        human_categories = ['crying_baby', 'sneezing', 'coughing', 'breathing', 'snoring', 'clapping']
        emergency_categories = ['siren', 'fire_alarm', 'chainsaw', 'helicopter']
        background_categories = ['rain', 'wind', 'thunderstorm', 'waves', 'crickets', 'frog', 'bird']
        
        # Read metadata if available
        if meta_file.exists():
            df = pd.read_csv(meta_file)
            for _, row in df.iterrows():
                audio_file = audio_dir / row['filename']
                category = row['category']
                
                if category in human_categories:
                    samples['human'].append(audio_file)
                elif category in emergency_categories:
                    samples['emergency'].append(audio_file)
                elif category in background_categories:
                    samples['background'].append(audio_file)
        else:
            # Fallback: categorize by filename
            for audio_file in audio_dir.glob("*.wav"):
                filename = audio_file.stem.lower()
                if any(cat in filename for cat in human_categories):
                    samples['human'].append(audio_file)
                elif any(cat in filename for cat in emergency_categories):
                    samples['emergency'].append(audio_file)
                elif any(cat in filename for cat in background_categories):
                    samples['background'].append(audio_file)
        
        print(f"  Added ESC-50: {len(samples['human'])} human, {len(samples['emergency'])} emergency, {len(samples['background'])} background")
        return samples
    
    def process_urban_sound(self, urban_path, samples):
        """Process UrbanSound8K dataset with categorization"""
        audio_dir = urban_path / "audio"
        meta_file = urban_path / "metadata" / "UrbanSound8K.csv"
        
        if not meta_file.exists():
            return samples
        
        # UrbanSound8K class mapping
        human_classes = ['children_playing', 'street_music']
        emergency_classes = ['siren', 'car_horn']
        background_classes = ['air_conditioner', 'drilling', 'jackhammer', 'dog_bark', 'engine_idling']
        
        df = pd.read_csv(meta_file)
        
        for _, row in df.iterrows():
            fold = f"fold{row['fold']}"
            audio_file = audio_dir / fold / row['slice_file_name']
            class_name = row['class']
            
            if audio_file.exists():
                if class_name in human_classes:
                    samples['human'].append(audio_file)
                elif class_name in emergency_classes:
                    samples['emergency'].append(audio_file)
                elif class_name in background_classes:
                    samples['background'].append(audio_file)
        
        print(f"  Added UrbanSound8K: {len(samples['human'])} human, {len(samples['emergency'])} emergency, {len(samples['background'])} background")
        return samples
    
    def balance_classes(self, samples, target_per_class=2000):
        """Balance classes with oversampling if needed"""
        balanced = {}
        
        # Find minimum available samples to set realistic target
        available_samples = {cls: len(samples[cls]) for cls in samples}
        min_available = min(available_samples.values())
        realistic_target = min(target_per_class, min_available)
        
        print(f"\nBalancing classes (target: {realistic_target} per class)...")
        
        for class_name, class_samples in samples.items():
            print(f"  {class_name}: {len(class_samples)} available")
            
            if len(class_samples) >= realistic_target:
                balanced[class_name] = random.sample(class_samples, realistic_target)
            else:
                # Use all available if we don't have enough
                balanced[class_name] = class_samples
                print(f"    Warning: Using only {len(class_samples)} samples")
        
        return balanced
    
    def split_and_save(self, balanced_samples, train_ratio=0.8):
        """Split samples into train/val and save"""
        print("\nSplitting and saving dataset...")
        
        for class_name, samples in balanced_samples.items():
            if not samples:
                continue
                
            random.shuffle(samples)
            split_idx = int(len(samples) * train_ratio)
            
            # Save train samples
            for i, sample_path in enumerate(samples[:split_idx]):
                dest = self.processed_dir / "train" / class_name / f"{class_name}_{i:06d}.wav"
                shutil.copy2(sample_path, dest)
            
            # Save val samples
            for i, sample_path in enumerate(samples[split_idx:]):
                dest = self.processed_dir / "val" / class_name / f"{class_name}_{i:06d}.wav"
                shutil.copy2(sample_path, dest)
            
            print(f"  {class_name}: {split_idx} train, {len(samples) - split_idx} val")
    
    def print_dataset_stats(self):
        """Print final dataset statistics"""
        print("\n=== DATASET STATISTICS ===")
        
        for split in ['train', 'val']:
            print(f"\n{split.upper()} Split:")
            total = 0
            for class_name in ['human', 'emergency', 'background']:
                count = len(list((self.processed_dir / split / class_name).glob("*.wav")))
                print(f"  {class_name}: {count} samples")
                total += count
            print(f"  TOTAL: {total} samples")

if __name__ == "__main__":
    preparer = DisasterAudioPreparer()
    preparer.prepare_dataset()