# src/models/thermal_head.py
import os, json, argparse, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

class ThermalHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)

def train_and_eval(feat_dir: str, out_dir: str, epochs=50, lr=1e-3):
    os.makedirs(out_dir, exist_ok=True)
    tr = np.load(os.path.join(feat_dir, "train.npz"))
    va = np.load(os.path.join(feat_dir, "val.npz"))
    with open(os.path.join(feat_dir, "classes.json")) as f: classes = json.load(f)

    Xtr = torch.tensor(tr["X"]).float(); ytr = torch.tensor(tr["y"]).long()
    Xva = torch.tensor(va["X"]).float(); yva = torch.tensor(va["y"]).long()

    model = ThermalHead(Xtr.shape[1], len(classes))
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best, best_state = 0.0, None

    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(Xtr), ytr); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(Xva); acc = (logits.argmax(1) == yva).float().mean().item()
        if acc > best: best, best_state = acc, {k:v.clone() for k,v in model.state_dict().items()}
        if (ep+1) % 10 == 0: print(f"Epoch {ep+1}/{epochs} val_acc={acc:.3f}")
    if best_state: model.load_state_dict(best_state)

    torch.save(model.state_dict(), os.path.join(out_dir, "thermal_head.pt"))
    with torch.no_grad():
        logits = model(Xva).numpy()
    yhat = logits.argmax(1)
    rep = classification_report(yva.numpy(), yhat, target_names=classes, output_dict=True, zero_division=0)
    cm  = confusion_matrix(yva.numpy(), yhat).tolist()
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"accuracy": rep["accuracy"], "report": rep, "classes": classes, "confusion_matrix": cm}, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, "val_logits.npz"), logits=logits, labels=yva.numpy())
    print(f"Saved model+metrics to {out_dir} | val_acc={rep['accuracy']:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", default="data/processed/thermal_features")
    ap.add_argument("--out_dir", default="data/processed/thermal")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    train_and_eval(a.feat_dir, a.out_dir, a.epochs, a.lr)