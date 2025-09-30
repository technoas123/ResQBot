import os, glob, json, argparse
import numpy as np, pandas as pd

def features_from_window(amb: np.ndarray, obj: np.ndarray) -> np.ndarray:
    d = obj - amb
    x = []
    x += [obj.mean(), obj.std(), obj.min(), obj.max(), np.percentile(obj,5), np.percentile(obj,95)]
    x += [d.mean(), d.std(), d.min(), d.max()]
    idx = np.arange(len(obj))
    slope_o = np.polyfit(idx, obj, 1)[0] if len(obj)>1 else 0.0
    slope_d = np.polyfit(idx, d, 1)[0] if len(d)>1 else 0.0
    x += [slope_o, slope_d, obj[-1], d[-1]]
    return np.array(x, dtype=np.float32)  # 16 features

def window_session(df: pd.DataFrame, win_sec: float, step_sec: float):
    t = df["t"].astype(float).values
    amb = df["ambient_c"].astype(float).values
    obj = df["object_c"].astype(float).values
    dt = np.median(np.diff(t)) if len(t)>1 else 0.1
    win = max(1, int(round(win_sec/dt)))
    step = max(1, int(round(step_sec/dt)))
    feats = []
    for i in range(0, len(obj)-win+1, step):
        feats.append(features_from_window(amb[i:i+win], obj[i:i+win]))
    return np.stack(feats) if feats else np.empty((0,16), dtype=np.float32)

def build_split(raw_root: str, split: str, out_dir: str, win_sec=2.0, step_sec=1.0):
    X_list, y_list = [], []
    split_dir = os.path.join(raw_root, split)
    classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    for ci, cls in enumerate(classes):
        for fp in sorted(glob.glob(os.path.join(split_dir, cls, "*.csv"))):
            df = pd.read_csv(fp)
            Xw = window_session(df, win_sec, step_sec)
            if len(Xw):
                X_list.append(Xw); y_list.append(np.full(len(Xw), ci, np.int64))
    if not X_list:
        raise RuntimeError(f"No windows built for split={split}")
    X = np.concatenate(X_list); y = np.concatenate(y_list)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{split}.npz"), X=X, y=y)
    with open(os.path.join(out_dir, "classes.json"), "w") as f: json.dump(classes, f)
    print(split, "X:", X.shape, "classes:", classes)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default="data/raw/thermal")
    ap.add_argument("--out_dir", default="data/processed/thermal_features")
    ap.add_argument("--win_sec", type=float, default=2.0)
    ap.add_argument("--step_sec", type=float, default=1.0)
    a = ap.parse_args()
    for s in ["train", "val"]:
        build_split(a.raw_root, s, a.out_dir, a.win_sec, a.step_sec)