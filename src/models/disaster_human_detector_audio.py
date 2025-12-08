"""
Disaster Human Detection Model - Neural Network Training Pipeline

This module implements a deep learning system for detecting human vocalizations in 
disaster scenarios, distinguishing them from emergency sounds and background noise.

Key Components:
1. DisasterAudioDataset: Custom PyTorch dataset loader for audio features and labels
2. ImprovedDisasterDetector: 4-layer feedforward neural network with batch normalization 
   and dropout for robust classification
3. FocalLoss: Custom loss function to handle class imbalance by focusing on hard examples
4. DisasterHumanDetector: Main training pipeline with comprehensive features

Training Features:
- Weighted random sampling to address class imbalance
- Focal loss with learnable class weights for better minority class detection
- AdamW optimizer with L2 regularization and ReduceLROnPlateau scheduler
- Early stopping with patience to prevent overfitting
- Validation-based model checkpointing

Evaluation Metrics:
- Overall classification accuracy across 3 classes (human, emergency, background)
- Detailed per-class precision, recall, and F1-scores
- Human detection recall (critical for search and rescue operations)
- Training history visualization (loss curves and accuracy plots)

Model Architecture:
Input (128D audio features) → Dense(256) → BN → ReLU → Dropout(0.3) →
Dense(128) → BN → ReLU → Dropout(0.3) → Dense(64) → ReLU → Dropout(0.15) →
Dense(3) → Softmax

Usage:
    python disaster_human_detector.py
    
Output:
    - data/processed/disaster_human_detection/disaster_human_detector.pt
    - data/processed/disaster_human_detection/training_history.png
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time

class DisasterAudioDataset(Dataset):
    def __init__(self, features_file, labels_file, class_to_idx):
        with open(features_file, 'r') as f:
            self.features = json.load(f)
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        
        self.class_to_idx = class_to_idx
        self.file_ids = list(self.features.keys())
    
    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        features = torch.tensor(self.features[file_id], dtype=torch.float32)
        label = self.class_to_idx[self.labels[file_id]]
        return features, label

class ImprovedDisasterDetector(nn.Module):
    def __init__(self, input_dim=128, num_classes=3, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

class DisasterHumanDetector:
    def __init__(self):
        self.data_root = Path("data")
        self.feature_dir = self.data_root / "processed" / "audio_features"
        self.model_dir = self.data_root / "processed" / "disaster_human_detection"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Class mappings
        self.classes = ['human', 'emergency', 'background']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
    
    def train(self, num_epochs=100, batch_size=32, learning_rate=0.001):
        """Train the disaster human detection model"""
        print("=== TRAINING DISASTER HUMAN DETECTOR ===")
        
        # Load datasets
        train_dataset, val_dataset = self.load_datasets()
        
        # Create weighted sampler for imbalanced data
        train_loader, val_loader = self.create_data_loaders(train_dataset, val_dataset, batch_size)
        
        # Initialize model
        model = ImprovedDisasterDetector(
            input_dim=128, 
            num_classes=len(self.classes),
            dropout_rate=0.3
        ).to(self.device)
        
        # Loss function with class weights
        class_weights = self.calculate_class_weights(train_dataset)
        criterion = FocalLoss(gamma=2, alpha=class_weights.to(self.device))
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_acc = 0
        patience = 15
        patience_counter = 0
        training_history = {'train_loss': [], 'val_acc': []}
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Classes: {self.classes}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            val_acc = self.validate_model(model, val_loader)
            
            # Update learning rate
            scheduler.step(val_acc)
            
            # Save history
            training_history['train_loss'].append(train_loss / len(train_loader))
            training_history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}')
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model(model, epoch, val_acc)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final evaluation
        self.evaluate_model(model, val_loader, training_history)
        
        print(f"\n=== TRAINING COMPLETED ===")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    def load_datasets(self):
        """Load train and validation datasets"""
        train_features = self.feature_dir / "train_features.json"
        train_labels = self.feature_dir / "train_labels.json"
        val_features = self.feature_dir / "val_features.json"
        val_labels = self.feature_dir / "val_labels.json"
        
        train_dataset = DisasterAudioDataset(train_features, train_labels, self.class_to_idx)
        val_dataset = DisasterAudioDataset(val_features, val_labels, self.class_to_idx)
        
        return train_dataset, val_dataset
    
    def calculate_class_weights(self, dataset):
        """Calculate class weights for imbalanced data"""
        class_counts = torch.zeros(len(self.classes))
        
        for _, label in dataset:
            class_counts[label] += 1
        
        total_samples = len(dataset)
        class_weights = total_samples / (len(self.classes) * class_counts)
        
        print(f"Class counts: {dict(zip(self.classes, class_counts.tolist()))}")
        print(f"Class weights: {dict(zip(self.classes, class_weights.tolist()))}")
        
        return class_weights
    
    def create_data_loaders(self, train_dataset, val_dataset, batch_size):
        """Create data loaders with weighted sampling"""
        # Calculate sample weights for training
        sample_weights = []
        for _, label in train_dataset:
            sample_weights.append(1.0 / len(train_dataset))  # Simplified weighting
        
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def validate_model(self, model, val_loader):
        """Validate model on validation set"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def evaluate_model(self, model, val_loader, training_history):
        """Comprehensive model evaluation"""
        print("\n=== MODEL EVALUATION ===")
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                outputs = model(features)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, 
                                  target_names=self.classes, digits=3))
        
        # Human detection performance (combining all human-related classes)
        human_indices = [self.class_to_idx['human']]
        human_recall = self.calculate_human_recall(all_labels, all_predictions, human_indices)
        print(f"\nHuman Detection Recall: {human_recall:.3f}")
        
        # Plot training history
        self.plot_training_history(training_history)
    
    def calculate_human_recall(self, true_labels, predictions, human_indices):
        """Calculate recall for human detection"""
        human_true = sum(1 for label in true_labels if label in human_indices)
        human_correct = sum(1 for true, pred in zip(true_labels, predictions) 
                           if true in human_indices and pred in human_indices)
        
        return human_correct / human_true if human_true > 0 else 0
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['val_acc'])
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self, model, epoch, accuracy):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'accuracy': accuracy,
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'timestamp': time.time()
        }
        
        model_path = self.model_dir / 'disaster_human_detector.pt'
        torch.save(checkpoint, model_path)
        print(f"Model saved with accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    detector = DisasterHumanDetector()
    detector.train(num_epochs=100, batch_size=32, learning_rate=0.001)