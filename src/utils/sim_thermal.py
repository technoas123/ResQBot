"""
Thermal Sensor Data Simulator for Human Detection Training

This module generates synthetic thermal sensor data mimicking MLX90614 (infrared 
thermometer) readings for training the thermal component of ResQBot's multi-modal
human detection system. Creates realistic time-series CSV files with ambient and
object temperature measurements.

Purpose:
Since collecting real thermal sensor data from disaster scenarios is challenging,
this simulator creates diverse, realistic training data with various environmental
conditions, sensor artifacts, and detection challenges that the model will face
in real-world deployment.

Simulated Classes:
1. HUMAN: 35-38°C signatures (4-6°C above ambient)
   - Natural body temperature fluctuations
   - Movement-induced signal variations
   - Partial occlusions (behind debris)
   
2. BACKGROUND: Ambient objects (-2°C to +4°C relative to ambient)
   - Cool surfaces, warm machinery, mixed environments
   - Overlapping temperature ranges for ML challenge
   
3. ANIMAL: Similar heat signatures (3-5°C above ambient)
   - Tests model's ability to distinguish humans from animals
   
4. MACHINERY: Hot equipment (2-8°C above ambient)
   - Cyclic heating patterns (engines, motors)
   - Different temporal characteristics than humans

Realistic Sensor Effects:
- Sensor noise: Gaussian noise σ=0.2°C (real sensors are noisy)
- Calibration drift: 0.01°C/second drift over time
- Measurement spikes: 1% probability of random outliers
- Sensor dropouts: 0.5% probability of reading failures
- Weather effects: Daily temperature cycles, environmental variations
- Distance attenuation: Signal strength reduces with distance (70-100%)
- Partial occlusions: 30% of sessions have temporary signal blockage

Dataset Generation:
- Default: 50 training + 20 validation sessions per class
- Duration: 10 seconds per session @ 10Hz sampling rate
- Output: CSV files with columns [t, ambient_c, object_c]

Temperature Ranges:
- Ambient: 15-35°C (various environments: indoor/outdoor, day/night)
- Object: Ambient ± delta (class-dependent)
- Clipped to physically realistic range: -10°C to 100°C

Output Structure:
data/raw/thermal/
├── train/
│   ├── human/session_human_XXX.csv
│   ├── background/session_background_XXX.csv
│   ├── animal/session_animal_XXX.csv
│   └── machinery/session_machinery_XXX.csv
└── val/ (same structure)

Usage:
    # Default dataset
    python script_name.py
    
    # Custom configuration
    python script_name.py --classes human background --train_n 100 --val_n 30 --dur 15

Output Format (CSV):
    t,ambient_c,object_c
    0.000,25.3456,31.2341
    0.100,25.3512,31.1987
    ...

Next Steps:
1. Feature extraction (compute statistics, gradients from CSV)
2. Model training (classify based on extracted features)
3. Real sensor integration (replace simulated data with MLX90614 readings)

Use Case: Provides diverse training data for thermal-based human detection,
enabling the model to handle real-world challenges like sensor noise, environmental
variations, and partial occlusions before deployment on rescue robots.
"""


import os, csv, time, argparse, random
from pathlib import Path
import numpy as np

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def rand_uniform(a,b): return random.random()*(b-a)+a

def simulate_series(label: str, dur: float, sr: float, seed: int=None):
    """
    Returns t, ambient_c, object_c arrays.
    Enhanced with realistic challenges for better ML training.
    """
    if seed is not None:
        np.random.seed(seed); random.seed(seed)

    n = int(dur * sr)
    t = np.arange(n, dtype=np.float64) / sr
    
    # More realistic ambient with weather effects
    base_amb = rand_uniform(15.0, 35.0)  # Wider range for different environments
    # Weather effects: sunny day, night cooling, etc.
    weather_effect = 5.0 * np.sin(2*np.pi * 0.002 * t + rand_uniform(0, 2*np.pi))
    amb_drift = 0.3 * np.sin(2*np.pi * rand_uniform(0.005, 0.02) * t + rand_uniform(0, 2*np.pi))
    amb_noise = np.random.normal(0, 0.1, n)  # Increased noise
    ambient = base_amb + weather_effect + amb_drift + amb_noise

    # Object delta with realistic human temperature variations
    delta = np.zeros(n, dtype=np.float64)
    
    if label.lower() in ["human", "hot"]:
        # REALISTIC HUMAN: 35-38°C with variations
        human_base = rand_uniform(4.0, 6.0)  # Above ambient
        # Human body temperature fluctuates
        human_variation = 0.8 * np.sin(2*np.pi * rand_uniform(0.05, 0.2) * t)
        # Movement causes changes in detected temperature
        movement_effect = 1.0 * np.sin(2*np.pi * rand_uniform(0.1, 0.5) * t)
        delta[:] = human_base + human_variation + movement_effect
        
    elif label.lower() in ["cool", "empty", "background"]:
        # Background objects: some slightly warm, some cool
        obj_type = random.choice(["cool", "warm_object", "mixed"])
        if obj_type == "cool":
            delta[:] = rand_uniform(-2.0, 1.0)  # Cooler than ambient
        elif obj_type == "warm_object":
            delta[:] = rand_uniform(1.0, 4.0)   # Warm but not human-level
        else:  # mixed
            delta[:] = rand_uniform(-1.0, 3.0)   # Overlapping with human range
        
    elif label.lower() == "animal":
        # Animals have similar but different heat signatures
        delta[:] = rand_uniform(3.0, 5.0)  # Slightly different range
        
    elif label.lower() == "machinery":
        # Machinery can be hot but with different patterns
        base = rand_uniform(2.0, 8.0)
        # Machinery often has cyclic heating patterns
        cycle = 2.0 * np.sin(2*np.pi * rand_uniform(0.02, 0.1) * t)
        delta[:] = base + cycle

    # REALISTIC SENSOR EFFECTS
    # Increased sensor noise (real sensors are noisy)
    obj_noise = np.random.normal(0, 0.2, n)  # More noise
    
    # Sensor calibration drift (real sensors drift over time)
    calibration_drift = 0.01 * t * np.random.choice([-1, 1])
    
    # Occasional sensor spikes
    spikes_mask = np.random.rand(n) < 0.01  # More frequent spikes
    spikes = spikes_mask * np.random.normal(0, 1.0, n)  # Larger spikes

    # Calculate object temperature FIRST
    object_c = ambient + delta + obj_noise + calibration_drift + spikes
    ambient_c = ambient

    # NOW add dropouts (using the calculated object_c)
    dropouts = np.random.rand(n) < 0.005  # Sensor dropouts
    dropout_effect = dropouts * (-object_c * 0.8)  # Sudden reading drops
    object_c += dropout_effect

    # REALISTIC ENVIRONMENTAL CHALLENGES
    # Partial occlusions (people behind partial barriers)
    if random.random() < 0.3:  # 30% of sessions have occlusions
        occlusion_start = np.random.randint(0, n//2)
        occlusion_duration = np.random.randint(int(0.3*sr), int(2.0*sr))
        occlusion_mask = np.zeros(n)
        occlusion_mask[occlusion_start:occlusion_start+occlusion_duration] = 1
        # Reduce signal during occlusion
        object_c -= occlusion_mask * delta * rand_uniform(0.3, 0.8)

    # Distance effects (signal strength decreases with distance)
    distance_factor = rand_uniform(0.7, 1.0)  # Simulate different distances
    object_c = ambient + (object_c - ambient) * distance_factor

    # Clip to reasonable range
    object_c = np.clip(object_c, -10, 100)
    ambient_c = np.clip(ambient_c, -10, 60)

    return t, ambient_c.astype(np.float32), object_c.astype(np.float32)

def write_session(out_csv, t, amb, obj):
    ensure_dir(Path(out_csv).parent)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t","ambient_c","object_c"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.3f}", f"{amb[i]:.4f}", f"{obj[i]:.4f}"])

def simulate_dataset(classes, train_n, val_n, dur, sr, out_root, seed=42):
    random.seed(seed); np.random.seed(seed)
    for split, count in [("train", train_n), ("val", val_n)]:
        for label in classes:
            out_dir = Path(out_root)/split/label
            ensure_dir(out_dir)
            for i in range(count):
                t, amb, obj = simulate_series(label, dur, sr, seed=None)
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_csv = out_dir/f"session_{label}_{i:03d}_{ts}.csv"
                write_session(out_csv, t, amb, obj)
        print(f"Wrote {count} sessions per class for split={split} -> {out_root}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["human","background","animal","machinery"], 
                   help="space-separated class names: human, background, animal, machinery")
    ap.add_argument("--train_n", type=int, default=50)
    ap.add_argument("--val_n", type=int, default=20)
    ap.add_argument("--dur", type=float, default=10.0, help="seconds per session")
    ap.add_argument("--sr", type=float, default=10.0, help="samples per second")
    ap.add_argument("--out_root", default="data/raw/thermal")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    simulate_dataset(args.classes, args.train_n, args.val_n, args.dur, args.sr, args.out_root, args.seed)