"""
Live Audio-Based Disaster Rescue Detection System

This module provides real-time audio monitoring and human voice detection for disaster
rescue operations. Continuously analyzes microphone input to identify human vocalizations,
emergency sirens, and background noise, triggering alerts when survivors are detected.

Core Functionality:
- Streams audio from selected microphone in real-time
- Processes 2-second rolling audio buffers
- Extracts 128-dimensional acoustic features (MFCCs, spectral, temporal)
- Classifies audio using trained disaster human detector neural network
- Triggers emergency alerts when human voices are detected

Detection Classes:
1. HUMAN: Voices, speech, cries for help, distress calls
   → Triggers "POTENTIAL SURVIVOR NEEDS RESCUE" alert
2. EMERGENCY: Sirens, alarms, emergency vehicle sounds
   → Triggers "EMERGENCY SERVICES ACTIVE" alert
3. BACKGROUND: Ambient noise, machinery, environmental sounds
   → Silent monitoring (no alert)

Alert Logic (Rescue-Optimized):
- Confidence threshold: 0.7 (default, adjustable)
- Consecutive detection requirement: 2+ frames for stability
- Alert debouncing: 3-second minimum between alerts (prevents spam)
- VU meter: Real-time audio level visualization
- Human detection prioritized for search and rescue operations

Audio Processing Pipeline:
1. Continuous audio streaming via sounddevice library
2. Circular buffer management (2-second rolling window)
3. Adaptive sample rate handling (8kHz - 48kHz compatibility)
4. Feature extraction matching training pipeline exactly:
   - Silence trimming (top_db=25)
   - Normalization
   - MFCCs (20 coefficients) + deltas + delta-deltas
   - Spectral centroid, rolloff, zero-crossing rate
   - Chroma features and RMS energy
5. Statistical aggregation (mean, std, max)
6. Neural network inference on GPU/CPU

Console Output:
- Live monitoring status with VU meter
- Real-time confidence scores for each detection
- Emergency alerts with distinct emoji indicators
- Session summary: total alerts, potential survivors count

Hardware Requirements:
- Any microphone (USB, built-in, professional audio interface)
- CPU/GPU for neural network inference
- Compatible sample rates: 8-48 kHz (auto-detected)

Command-Line Options:
    --model: Path to trained model checkpoint (.pt file)
    --device: Audio input device ID (prompts if not specified)
    --confidence: Detection threshold 0.0-1.0 (default: 0.7)
    --list-devices: Show available microphones and exit

Usage Examples:
    # Interactive mode (prompts for device)
    python live_disaster_tester.py
    
    # Specify device and confidence
    python live_disaster_tester.py --device 1 --confidence 0.75
    
    # List available microphones
    python live_disaster_tester.py --list-devices

Use Case Scenario:
Deploy on rescue robots or stationary monitoring stations in disaster zones.
Continuously scans environment for human survivors by detecting:
- Verbal cries for help
- Conversation (trapped people communicating)
- Distress vocalizations (moaning, coughing, breathing)
- Any human voice activity indicating presence of survivors

Alert Interpretation:
🚨 "HUMAN DETECTED" → High priority: Investigate location immediately
🚨 "SIREN/ALARM DETECTED" → Emergency services nearby, coordinate response
✅ Multiple human alerts → Confirms survivor presence, dispatch rescue team
❌ No alerts → Area clear or no survivors detected

Safety Features:
- Console encoding error handling (UTF-8 support for emoji alerts)
- Graceful Ctrl+C shutdown with session statistics
- Audio device compatibility checking and fallback
- Error recovery for audio dropouts and feature extraction failures

Integration:
Part of ResQBot's multi-modal detection system. Can be combined with:
- Vision model (YOLOv8 for visual confirmation)
- Thermal sensor (heat signature detection)
- Sensor fusion for higher confidence decisions

Performance:
- Processing latency: ~500ms per detection cycle
- Sample rate: Adaptive (typically 22.05 kHz)
- Buffer size: 2 seconds (balances responsiveness vs. accuracy)
- GPU acceleration: Automatic if available

Note: This is the AUDIO-ONLY standalone version. For full multi-modal fusion
(audio + thermal + vision), see the main prediction pipeline.
"""



import sys
import io
import argparse
import time
import numpy as np
import sounddevice as sd
import torch
import librosa
from pathlib import Path

def setup_console():
    """Fix console encoding issues"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

setup_console()

class ImprovedDisasterDetector(torch.nn.Module):
    def __init__(self, input_dim=128, num_classes=3, dropout_rate=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate * 0.5),
            
            torch.nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

class LiveDisasterTester:
    def __init__(self, model_path, confidence_threshold=0.6):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model, self.classes = self.load_model()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Audio settings
        self.sr = 22050
        self.buffer_size = int(self.sr * 2.0)  # 2-second buffer
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        
        # Detection history
        self.detection_history = []
        self.consecutive_detections = {cls: 0 for cls in self.classes}
        
        print(f"Loaded Disaster Human Detector")
        print(f"Classes: {', '.join(self.classes)}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Using device: {self.device}")
    
    def load_model(self):
        """Load trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Recreate the exact same model architecture
        model = ImprovedDisasterDetector(
            input_dim=128, 
            num_classes=len(checkpoint['classes']),
            dropout_rate=0.3
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, checkpoint['classes']
    
    def extract_features(self, audio, sr=22050):
        """Extract features matching training pipeline"""
        try:
            # Remove silence
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
            if len(audio_trimmed) < 512:
                audio_trimmed = audio
            
            # Normalize
            audio_trimmed = audio_trimmed / (np.max(np.abs(audio_trimmed)) + 1e-8)
            
            # Extract features (same as training)
            mfcc = librosa.feature.mfcc(y=audio_trimmed, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_trimmed, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_trimmed, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_trimmed)
            chroma = librosa.feature.chroma_stft(y=audio_trimmed, sr=sr)
            rms = librosa.feature.rms(y=audio_trimmed)
            
            all_features = np.vstack([
                mfcc, mfcc_delta, mfcc_delta2,
                spectral_centroid, spectral_rolloff,
                zero_crossing_rate, chroma, rms
            ])
            
            feature_means = np.mean(all_features, axis=1)
            feature_stds = np.std(all_features, axis=1)
            feature_max = np.max(all_features, axis=1)
            
            features = np.concatenate([feature_means, feature_stds, feature_max])
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            features = np.zeros(128)
        
        if len(features) < 128:
            features = np.pad(features, (0, 128 - len(features)))
        else:
            features = features[:128]
        
        return features.astype(np.float32)
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback"""
        if status:
            print(f"Audio status: {status}")
        
        audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
        chunk_size = len(audio_chunk)
        end_pos = self.write_pos + chunk_size
        
        if end_pos <= self.buffer_size:
            self.audio_buffer[self.write_pos:end_pos] = audio_chunk.flatten()
        else:
            first_part = self.buffer_size - self.write_pos
            self.audio_buffer[self.write_pos:] = audio_chunk[:first_part].flatten()
            self.audio_buffer[:chunk_size - first_part] = audio_chunk[first_part:].flatten()
        
        self.write_pos = (self.write_pos + chunk_size) % self.buffer_size
    
    def get_compatible_sample_rate(self, device_id, desired_rate=22050):
        """Find compatible sample rate for audio device"""
        try:
            device_info = sd.query_devices(device_id)
            supported_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000]
            
            # Check if desired rate is supported
            if desired_rate <= device_info['default_samplerate']:
                try:
                    with sd.InputStream(device=device_id, samplerate=desired_rate, channels=1):
                        pass
                    return desired_rate
                except:
                    pass
            
            # Find closest supported rate
            for rate in supported_rates:
                if rate <= device_info['default_samplerate']:
                    try:
                        with sd.InputStream(device=device_id, samplerate=rate, channels=1):
                            pass
                        print(f"Using compatible sample rate: {rate} Hz")
                        return rate
                    except:
                        continue
            
            # Fallback to default
            default_rate = int(device_info['default_samplerate'])
            print(f"Using device default sample rate: {default_rate} Hz")
            return default_rate
            
        except Exception as e:
            print(f"Error querying device: {e}")
            return 22050
    
    def list_audio_devices(self):
        """List available audio devices"""
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  {i}: {device['name']} (Input: {device['max_input_channels']} ch, Rate: {device['default_samplerate']} Hz)")
    
    def run_live_detection(self, device_id=None):
        """Run live disaster detection - HUMAN VOICE TRIGGERS EMERGENCY ALERT"""
        if device_id is None:
            self.list_audio_devices()
            try:
                device_id = int(input("\nEnter device ID: "))
            except ValueError:
                print("Invalid device ID")
                return
        
        # Get compatible sample rate
        self.sr = self.get_compatible_sample_rate(device_id, 22050)
        self.buffer_size = int(self.sr * 2.0)  # Reset buffer size
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        
        print(f"\n=== LIVE DISASTER RESCUE DETECTION STARTED ===")
        print(f"Sample rate: {self.sr} Hz")
        print(f"Buffer size: {self.buffer_size} samples ({self.buffer_size/self.sr:.1f} seconds)")
        print("🚨 ANY HUMAN VOICE WILL TRIGGER EMERGENCY ALERT 🚨")
        print("Perfect for disaster rescue operations!")
        print("Press Ctrl+C to stop\n")
        
        # Statistics
        human_emergency_alerts = 0
        siren_alerts = 0
        last_alert_time = time.time()
        
        try:
            with sd.InputStream(
                samplerate=self.sr, channels=1, dtype='float32',
                device=device_id, callback=self.audio_callback, blocksize=1024
            ):
                time.sleep(2.0)  # Warm-up time
                
                while True:
                    # Get current audio buffer
                    indices = np.arange(-self.buffer_size, 0) + self.write_pos
                    current_audio = self.audio_buffer[indices % self.buffer_size].copy()
                    
                    # Calculate audio level
                    audio_level = np.sqrt(np.mean(current_audio**2))
                    
                    # Only process if there's significant audio
                    if audio_level > 0.01:
                        # Resample if necessary
                        if self.sr != 22050:
                            current_audio_processed = librosa.resample(
                                current_audio, orig_sr=self.sr, target_sr=22050
                            )
                        else:
                            current_audio_processed = current_audio
                        
                        # Extract features and predict
                        features = self.extract_features(current_audio_processed, 22050)
                        audio_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(audio_tensor)
                            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                        
                        pred_idx = np.argmax(probabilities)
                        confidence = probabilities[pred_idx]
                        prediction = self.classes[pred_idx]
                        
                        # Update consecutive detections
                        for cls in self.classes:
                            if cls == prediction and confidence > self.confidence_threshold:
                                self.consecutive_detections[cls] += 1
                            else:
                                self.consecutive_detections[cls] = 0
                        
                        # VU meter
                        level_bars = "#" * min(int(audio_level * 50), 25)
                        spaces = " " * (25 - len(level_bars))
                        
                        # DISASTER RESCUE LOGIC: Human voice = Emergency Alert
                        current_time = time.time()
                        
                        if (prediction == 'human' and confidence > self.confidence_threshold and 
                            self.consecutive_detections['human'] >= 2 and
                            current_time - last_alert_time > 3.0):  # 3-second debouncing
                            
                            human_emergency_alerts += 1
                            last_alert_time = current_time
                            print(f"🚨 EMERGENCY: HUMAN DETECTED! #{human_emergency_alerts} (Confidence: {confidence:.3f})")
                            print(f"   ⚠️  POTENTIAL SURVIVOR NEEDS RESCUE!")
                        
                        elif (prediction == 'emergency' and confidence > self.confidence_threshold and 
                              self.consecutive_detections['emergency'] >= 2 and
                              current_time - last_alert_time > 3.0):
                            
                            siren_alerts += 1
                            last_alert_time = current_time
                            print(f"🚨 EMERGENCY: SIREN/ALARM DETECTED! #{siren_alerts} (Confidence: {confidence:.3f})")
                            print(f"   ⚠️  EMERGENCY SERVICES ACTIVE!")
                        
                        else:
                            status = f"Monitoring: [{level_bars}{spaces}] {prediction}: {confidence:.3f}"
                            print(status, end='\r')
                    
                    else:
                        # No significant audio
                        print("Monitoring: [                         ] Area Clear - Listening...", end='\r')
                    
                    time.sleep(0.5)  # Processing interval
                    
        except KeyboardInterrupt:
            print(f"\n\n=== RESCUE OPERATION SUMMARY ===")
            print(f"🚨 Human Emergency Alerts: {human_emergency_alerts}")
            print(f"🚨 Siren/Alarm Alerts: {siren_alerts}")
            print(f"📊 Total alerts: {human_emergency_alerts + siren_alerts}")
            if human_emergency_alerts > 0:
                print(f"✅ POTENTIAL SURVIVORS DETECTED: {human_emergency_alerts}")
            else:
                print(f"❌ No human survivors detected in this area")

def main():
    parser = argparse.ArgumentParser(description="Live Disaster Rescue Detection - HUMAN VOICE = EMERGENCY")
    parser.add_argument("--model", default="data/processed/disaster_human_detection/disaster_human_detector.pt")
    parser.add_argument("--device", type=int, default=None, help="Audio device ID")
    parser.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    
    args = parser.parse_args()
    
    if args.list_devices:
        tester = LiveDisasterTester(args.model, args.confidence)
        tester.list_audio_devices()
        return
    
    try:
        tester = LiveDisasterTester(args.model, args.confidence)
        tester.run_live_detection(args.device)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()