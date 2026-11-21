"""Loading the metadata, linking images, lesion-level splits and the resized image array."""
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .config import Config, CLASS_TO_IDX, IMAGE_DIRS


def find_image_paths(data_dir: Path) -> dict:
    """Map image_id -> file path, looking in the two HAM10000 folders (or any *.jpg below data_dir)."""
    folders = [data_dir / d for d in IMAGE_DIRS if (data_dir / d).exists()] or [data_dir]
    return {p.stem: p for f in folders for p in f.rglob("*.jpg")}


def load_metadata(cfg: Config) -> pd.DataFrame:
    meta = pd.read_csv(cfg.data_dir / "HAM10000_metadata.csv")
    paths = find_image_paths(cfg.data_dir)
    meta["path"] = meta["image_id"].map(paths)
    missing = int(meta["path"].isna().sum())
    if missing:
        raise FileNotFoundError(f"{missing} images listed in the metadata were not found under {cfg.data_dir}")
    meta["label"] = meta["dx"].map(CLASS_TO_IDX).astype(int)
    return meta


def _lesions(meta):
    return meta.drop_duplicates("lesion_id")[["lesion_id", "dx"]]


def assert_lesion_disjoint(meta, is_train):
    a, b = set(meta.loc[is_train, "lesion_id"]), set(meta.loc[~is_train, "lesion_id"])
    assert a.isdisjoint(b), "a lesion appears on both sides of the split"


def split_lesions_80_20(meta, seed=42, test_size=0.20):
    """Stratified 80/20 split by lesion. Returns a boolean mask: True = training image."""
    les = _lesions(meta)
    tr, _ = train_test_split(les, test_size=test_size, stratify=les["dx"], random_state=seed)
    is_train = meta["lesion_id"].isin(tr["lesion_id"]).values
    assert_lesion_disjoint(meta, is_train)
    return is_train


def split_lesions_70_15_15(meta, seed=42):
    """Stratified 70/15/15 split by lesion; returns an array of 'train' / 'val' / 'test'."""
    les = _lesions(meta)
    tr, tmp = train_test_split(les, test_size=0.30, stratify=les["dx"], random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp["dx"], random_state=seed)
    mapping = {**dict.fromkeys(tr.lesion_id, "train"), **dict.fromkeys(va.lesion_id, "val"), **dict.fromkeys(te.lesion_id, "test")}
    split = meta["lesion_id"].map(mapping).values
    assert_lesion_disjoint(meta, split == "train"); assert_lesion_disjoint(meta, split != "val")
    return split


def load_images(paths, size=128) -> np.ndarray:
    """Read every image once and resize it to size x size RGB (uint8)."""
    out = np.zeros((len(paths), size, size, 3), dtype=np.uint8)
    for i, p in enumerate(tqdm(paths, desc="resizing images")):
        out[i] = np.asarray(Image.open(p).convert("RGB").resize((size, size)))
    return out


def prepare(cfg: Config, seed=42):
    """Build and save the processed image array and metadata table (both splits)."""
    meta = load_metadata(cfg)
    meta["split"] = split_lesions_70_15_15(meta, seed)               # deep-learning study
    meta["split80"] = np.where(split_lesions_80_20(meta, seed), "train", "test")   # Phase 1
    imgs = load_images(meta["path"], cfg.img_size)
    np.save(cfg.processed_dir / f"images_{cfg.img_size}.npy", imgs)
    meta.drop(columns=["path"]).to_csv(cfg.processed_dir / "meta_processed.csv", index=False)
    return meta, imgs


def load_processed(cfg: Config):
    """Load what prepare() saved (images are memory-mapped, not read into RAM)."""
    imgs = np.load(cfg.processed_dir / f"images_{cfg.img_size}.npy", mmap_mode="r")
    meta = pd.read_csv(cfg.processed_dir / "meta_processed.csv")
    return meta, imgs
