"""Fast unit tests of the building blocks (a few seconds in total)."""
import numpy as np, pandas as pd
from ham10000.config import Config, CLASSES
from ham10000.data import load_metadata, split_lesions_80_20, split_lesions_70_15_15
from ham10000.features import image_features, metadata_matrix, IMG_FEATURES, feature_group
from ham10000.metaheuristics import make_fitness, penalised, run_ga, run_bpso, run_nsga2, knee, nd_sort
from ham10000.reduction import sne
from ham10000.synthetic import make_synthetic_dataset


def _meta(tmp_path, n=250):
    make_synthetic_dataset(tmp_path / "raw", n_lesions=n)
    return load_metadata(Config(data_dir=tmp_path / "raw", work_dir=tmp_path / "w", results_dir=tmp_path / "r", figures_dir=tmp_path / "f"))


def test_splits_never_share_a_lesion(tmp_path):
    meta = _meta(tmp_path)
    is_tr = split_lesions_80_20(meta, seed=1)
    assert set(meta.loc[is_tr, "lesion_id"]).isdisjoint(meta.loc[~is_tr, "lesion_id"])
    split = split_lesions_70_15_15(meta, seed=1)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert set(meta.loc[split == a, "lesion_id"]).isdisjoint(meta.loc[split == b, "lesion_id"])
    assert set(split) == {"train", "val", "test"}


def test_age_imputation_uses_training_median_only(tmp_path):
    meta = _meta(tmp_path)
    is_tr = np.zeros(len(meta), bool); is_tr[: len(meta) // 2] = True
    meta.loc[~is_tr, "age"] = 999.0                      # the test half must not influence the imputation
    meta.loc[meta.index[0], "age"] = np.nan
    X, names = metadata_matrix(meta, is_tr)
    assert names[0] == "age" and not np.isnan(X).any()
    assert X[0, 0] == meta.loc[is_tr, "age"].median()


def test_image_features_shape_and_names():
    rng = np.random.default_rng(0); imgs = rng.integers(0, 256, (6, 128, 128, 3), dtype=np.uint8)
    F = image_features(imgs)
    assert F.shape == (6, len(IMG_FEATURES)) == (6, 28) and np.isfinite(F).all()
    assert {feature_group(n) for n in IMG_FEATURES} == {"RGB stats", "HSV", "centre/border", "texture"}


def test_metaheuristics_return_valid_masks():
    rng = np.random.default_rng(0); n, d = 240, 12
    y = rng.integers(0, 3, n); g = np.repeat(np.arange(n // 2), 2); X = rng.normal(size=(n, d)); X[:, 0] += y * 2
    f1_of = make_fitness(X, y, g, seed=0, n_trees=8); fit = penalised(f1_of)
    best, hist, mean_hist = run_ga(d, fit, pop=6, gens=3, seed=0)
    assert best.dtype == bool and best.any() and len(hist) == len(mean_hist) == 3 and hist[-1] >= hist[0]
    gb, h2 = run_bpso(d, fit, pop=6, iters=3, seed=0); assert gb.any() and len(h2) == 3
    M, f1s, sizes, h3 = run_nsga2(d, f1_of, pop=6, gens=3, seed=0)
    assert len(M) == len(f1s) == len(sizes) and (sizes == M.sum(1)).all()
    assert knee(M, f1s, sizes).sum() == sizes[np.where(f1s >= f1s.max() - 0.01)[0]].min()
    assert best[0] or gb[0] or M[:, 0].any()           # the informative feature is picked up by at least one method


def test_nondominated_sort_puts_dominated_points_last():
    F = np.array([[1, 5], [2, 2], [5, 1], [3, 3], [6, 6]])
    fronts = nd_sort(F)
    assert set(fronts[0]) == {0, 1, 2} and fronts[-1] == [4]


def test_sne_runs_and_is_finite():
    rng = np.random.default_rng(0); X = np.vstack([rng.normal(0, 1, (40, 5)), rng.normal(6, 1, (40, 5))])
    Y = sne(X, perplexity=10, n_iter=150)
    assert Y.shape == (80, 2) and np.isfinite(Y).all()
    assert np.linalg.norm(Y[:40].mean(0) - Y[40:].mean(0)) > 0.5      # the two groups are separated
