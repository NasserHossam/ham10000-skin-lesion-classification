"""Feature tables: flattened pixels, the 28 colour/texture features and the metadata features."""
import numpy as np, pandas as pd
from PIL import Image
from matplotlib.colors import rgb_to_hsv
from scipy.stats import skew

IMG_FEATURES = ([f"{c}_mean" for c in "RGB"] + [f"{c}_std" for c in "RGB"] + [f"{c}_skew" for c in "RGB"]
                + ["H_mean", "S_mean", "V_mean", "H_std", "S_std", "V_std"]
                + [f"{c}_center" for c in "RGB"] + [f"{c}_border" for c in "RGB"]
                + [f"{c}_center_minus_border" for c in "RGB"]
                + ["grad_mean", "grad_std", "laplacian_var", "gray_entropy"])

FEATURE_GROUPS = {"RGB stats": "RGB statistics", "HSV": "HSV statistics", "centre/border": "centre vs border",
                  "texture": "texture", "age": "age", "sex": "sex", "location": "location"}


def feature_group(name: str) -> str:
    """Group of a feature, used by the ablation experiment."""
    if name.startswith("sex_"): return "sex"
    if name.startswith("localization_"): return "location"
    if name == "age": return "age"
    if name.endswith(("_center_minus_border", "_center", "_border")): return "centre/border"
    if name.split("_")[0] in ("H", "S", "V"): return "HSV"
    if name.split("_")[0] in ("R", "G", "B"): return "RGB stats"
    return "texture"


def flat_features(images, size, gray=False) -> np.ndarray:
    """Resize every image to size x size and flatten it into one row of values in [0, 1]."""
    out = []
    for a in images:
        im = Image.fromarray(np.asarray(a))
        if gray: im = im.convert("L")
        out.append(np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32).ravel() / 255)
    return np.stack(out)


def _extract(batch) -> np.ndarray:
    f = np.asarray(batch).astype(np.float32) / 255; n = len(f)
    mean, std = f.mean((1, 2)), f.std((1, 2))
    sk = np.nan_to_num(skew(f.reshape(n, -1, 3), axis=1))
    hsv = rgb_to_hsv(f); hm, hs = hsv.mean((1, 2)), hsv.std((1, 2))
    h = f.shape[1]; a, b = h // 4, 3 * h // 4                       # central half of the image
    cen = f[:, a:b, a:b].mean((1, 2))
    bor = (f.sum((1, 2)) - f[:, a:b, a:b].sum((1, 2))) / (h * h - (b - a) ** 2)
    g = f.mean(3)
    gx = np.diff(g, axis=2)[:, :-1, :]; gy = np.diff(g, axis=1)[:, :, :-1]
    grad = np.sqrt(gx ** 2 + gy ** 2)
    lap = 4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]
    ent = np.zeros(n, np.float32)
    for i in range(n):
        p = np.bincount((g[i] * 31).astype(int).ravel(), minlength=32) / g[i].size
        p = p[p > 0]; ent[i] = -(p * np.log2(p)).sum()
    return np.column_stack([mean, std, sk, hm, hs, cen, bor, cen - bor,
                            grad.mean((1, 2)), grad.std((1, 2)), lap.var((1, 2)), ent]).astype(np.float32)


def image_features(images, batch=500) -> np.ndarray:
    """The 28 interpretable image features (colour statistics, centre/border contrast, texture)."""
    return np.vstack([_extract(images[i:i + batch]) for i in range(0, len(images), batch)])


def metadata_matrix(meta: pd.DataFrame, is_train) -> tuple:
    """Age (missing values imputed with the TRAIN median) plus one-hot sex and location."""
    age = meta["age"].fillna(meta.loc[is_train, "age"].median()).values
    dummies = pd.get_dummies(meta[["sex", "localization"]], dtype=int)
    return np.column_stack([age, dummies.values]).astype(np.float32), ["age"] + list(dummies.columns)


def build_feature_table(images, meta, is_train):
    """The 47-feature table (28 image + 19 metadata) and its feature names."""
    Xm, mnames = metadata_matrix(meta, is_train)
    return np.hstack([image_features(images), Xm]), IMG_FEATURES + mnames
