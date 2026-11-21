"""Tiny fake HAM10000-like dataset, so the whole pipeline can be tested without the real images."""
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from .config import CLASSES, IMAGE_DIRS

LOCATIONS = ["abdomen", "back", "chest", "ear", "face", "foot", "hand", "lower extremity", "neck", "scalp", "trunk", "unknown", "upper extremity"]


def make_synthetic_dataset(data_dir, n_lesions=280, seed=0, size=96):
    """Write HAM10000_metadata.csv and JPEGs in the same layout as the real dataset (some lesions have 2 images)."""
    rng = np.random.default_rng(seed); data_dir = Path(data_dir)
    for d in IMAGE_DIRS: (data_dir / d).mkdir(parents=True, exist_ok=True)
    probs = np.array([.05, .08, .12, .04, .13, .50, .08]); rows = []
    for k in range(n_lesions):
        c = rng.choice(len(CLASSES), p=probs); n_img = 1 + int(rng.random() < 0.3)
        age = float(rng.choice(np.arange(20, 85, 5))) if rng.random() > 0.03 else np.nan
        sex, loc = rng.choice(["female", "male"]), rng.choice(LOCATIONS)
        for j in range(n_img):
            iid = f"ISIC_{k:05d}{j}"
            base = np.array([200, 150, 140]) - 12 * c * np.array([1, .8, .5])          # class-dependent colour
            img = np.clip(base + rng.normal(0, 18, (size, size, 3)), 0, 255)
            yy, xx = np.mgrid[:size, :size]; mask = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size / 4) ** 2
            img[mask] *= 0.6 + 0.05 * c                                                  # darker lesion in the centre
            Image.fromarray(img.astype(np.uint8)).save(data_dir / IMAGE_DIRS[j % 2] / f"{iid}.jpg")
            rows.append(dict(lesion_id=f"HAM_{k:07d}", image_id=iid, dx=CLASSES[c], dx_type="histo", age=age, sex=sex, localization=loc))
    pd.DataFrame(rows).to_csv(data_dir / "HAM10000_metadata.csv", index=False)
    return data_dir
