"""Shared loading helpers used by several scripts."""
import numpy as np, pandas as pd
from .config import Config
from .data import load_processed
from .features import build_feature_table
from .reduction import scale_on_train


def handcrafted_data(cfg: Config) -> dict:
    """The 47-feature table for the 80/20 split: raw, scaled with train statistics, names, labels and lesion groups."""
    meta, imgs = load_processed(cfg)
    is_train = (meta["split80"] == "train").values
    path = cfg.processed_dir / "features_handcrafted.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True); X, names = d["X"], [str(n) for n in d["names"]]
    else:
        X, names = build_feature_table(imgs, meta, is_train)
        np.savez_compressed(path, X=X, names=np.array(names)); print("saved", path)
    Xs, _ = scale_on_train(X, is_train)
    return dict(meta=meta, X=X, Xs=Xs, names=names, is_train=is_train, y=meta["label"].values, groups=meta["lesion_id"].values)


def save_table(df: pd.DataFrame, path, index=False):
    df.to_csv(path, index=index); print("saved", path)
