"""Deep-learning reference study: ResNet18 on images, and a fusion of image and metadata (needs PyTorch).

Run on a GPU if you have an NVIDIA card; on CPU it works but is slow (use --epochs 2 --limit 2000 to try it).
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from .config import Config, CLASSES, MEL
from .features import metadata_matrix


def _torch():
    try:
        import torch
    except ImportError as e:
        raise SystemExit("PyTorch is required for the deep-learning part: pip install torch torchvision") from e
    return torch


def metadata_baseline(meta):
    """Random Forest on age, sex and location only."""
    is_tr = (meta["split"] == "train").values
    X, _ = metadata_matrix(meta, is_tr); y = meta["label"].values
    rf = RandomForestClassifier(300, class_weight="balanced", random_state=42, n_jobs=-1).fit(X[is_tr], y[is_tr])
    te = (meta["split"] == "test").values
    return rf.predict(X[te]), y[te]


def train_and_evaluate(imgs, meta, cfg: Config, epochs=10, batch_size=64, num_workers=0, limit=None):
    """Train ResNet18 and the fusion model on the 70/15/15 split; evaluate the best epoch of each on the test split."""
    torch = _torch(); import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms as T
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("Device:", dev)
    y_all = meta["label"].values
    idx = {s: np.where(meta["split"] == s)[0] for s in ["train", "val", "test"]}
    if limit: idx = {s: v[:limit] for s, v in idx.items()}
    is_tr = (meta["split"] == "train").values
    M, _ = metadata_matrix(meta, is_tr); mu, sd = M[idx["train"], 0].mean(), M[idx["train"], 0].std() or 1.0; M[:, 0] = (M[:, 0] - mu) / sd
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    aug = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20), T.ColorJitter(0.2, 0.2, 0.2)])

    class DS(Dataset):
        def __init__(self, ids, train, with_meta): self.ids, self.train, self.with_meta = ids, train, with_meta
        def __len__(self): return len(self.ids)
        def __getitem__(self, i):
            j = self.ids[i]; x = torch.from_numpy(np.array(imgs[j])).permute(2, 0, 1).float() / 255
            if self.train: x = aug(x)
            x = (x - mean) / std
            return (x, torch.from_numpy(M[j]), int(y_all[j])) if self.with_meta else (x, int(y_all[j]))

    def loaders(with_meta):
        return {s: DataLoader(DS(idx[s], s == "train", with_meta), batch_size=batch_size, shuffle=(s == "train"), num_workers=num_workers) for s in idx}

    class Fusion(nn.Module):
        def __init__(self, nmeta):
            super().__init__()
            self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT); self.cnn.fc = nn.Identity()
            self.meta = nn.Sequential(nn.Linear(nmeta, 64), nn.ReLU(), nn.Dropout(0.2))
            self.head = nn.Sequential(nn.Linear(512 + 64, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 7))
        def forward(self, x, m): return self.head(torch.cat([self.cnn(x), self.meta(m)], 1))

    w = compute_class_weight("balanced", classes=np.arange(7), y=y_all[idx["train"]]) if len(np.unique(y_all[idx["train"]])) == 7 else np.ones(7)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float).to(dev))

    def run(model, name, with_meta):
        ld = loaders(with_meta); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4); best, path = -1, cfg.processed_dir / f"{name}_best.pt"
        fwd = (lambda b: model(b[0].to(dev), b[1].to(dev))) if with_meta else (lambda b: model(b[0].to(dev)))
        def predict(split):
            model.eval(); P, Y = [], []
            with torch.no_grad():
                for b in ld[split]: P.append(fwd(b).argmax(1).cpu()); Y.append(b[-1])
            return torch.cat(Y).numpy(), torch.cat(P).numpy()
        for ep in range(epochs):
            model.train(); tot = 0
            for b in ld["train"]:
                opt.zero_grad(); loss = crit(fwd(b), b[-1].to(dev)); loss.backward(); opt.step(); tot += loss.item() * len(b[-1])
            yt, yp = predict("val"); f1 = f1_score(yt, yp, average="macro")
            print(f"{name} epoch {ep + 1}: loss {tot / len(idx['train']):.4f} | val macro-F1 {f1:.4f}")
            if f1 > best: best = f1; torch.save(model.state_dict(), path)
        model.load_state_dict(torch.load(path)); return predict("test")

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT); resnet.fc = nn.Linear(512, 7)
    y_img, p_img = run(resnet.to(dev), "resnet18", False)
    y_fus, p_fus = run(Fusion(M.shape[1]).to(dev), "fusion", True)
    p_meta, y_meta = metadata_baseline(meta)
    rows = []
    for name, y, p in [("Metadata RF", y_meta, p_meta), ("ResNet18", y_img, p_img), ("Fusion", y_fus, p_fus)]:
        rows.append({"Model": name, "Accuracy": accuracy_score(y, p), "Macro-F1": f1_score(y, p, average="macro"),
                     "Melanoma recall": recall_score(y, p, labels=[MEL], average=None)[0]})
    return pd.DataFrame(rows).round(3)
