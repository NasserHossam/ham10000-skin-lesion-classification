"""Filter, wrapper and embedded feature selection (Phase 1, Steps 4 and 5)."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


def filter_scores(Xtr, ytr, seed=42):
    """Mutual information and ANOVA F-score of every feature (training data only)."""
    return mutual_info_classif(Xtr, ytr, random_state=seed), f_classif(Xtr, ytr)[0]


def top_k(scores, k):
    return list(np.argsort(scores)[::-1][:k])


def wrapper_select(Xtr, ytr, gtr, k, direction, sub=2500, seed=0):
    """Sequential forward/backward selection with logistic regression and lesion-grouped 3-fold CV."""
    s = np.random.default_rng(seed).choice(len(Xtr), min(sub, len(Xtr)), replace=False)
    cv = list(GroupKFold(3).split(Xtr[s], ytr[s], gtr[s]))
    sfs = SequentialFeatureSelector(LogisticRegression(max_iter=300, class_weight="balanced"), n_features_to_select=k,
                                    direction=direction, scoring="f1_macro", cv=cv, n_jobs=-1).fit(Xtr[s], ytr[s])
    return list(np.where(sfs.get_support())[0])


def _lr_penalised(kind, C, seed):
    kw = dict(solver="saga", C=C, max_iter=300, tol=1e-3, class_weight="balanced", random_state=seed)
    if kind == "L1": return LogisticRegression(penalty="l1", **kw)
    if kind == "L2": return LogisticRegression(penalty="l2", **kw)
    return LogisticRegression(penalty="elasticnet", l1_ratio=0.5, **kw)


def embedded_select(Xtr, ytr, kind="L1", max_feats=25, seed=42, verbose=True):
    """Largest regularisation strength that keeps at most max_feats features (a feature is kept if non-zero for any class)."""
    chosen = None
    for C in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
        m = _lr_penalised(kind, C, seed).fit(Xtr, ytr)
        nz = np.where(np.abs(m.coef_).max(0) > 1e-6)[0]
        if verbose: print(f"{kind}  C={C}: {len(nz)} features kept")
        if chosen is None or len(nz) <= max_feats: chosen = (C, m, nz)
    return chosen


def l2_top_k(Xtr, ytr, k, C=0.05, seed=42):
    """L2 never zeroes coefficients, so keep the k features with the largest mean |coefficient|."""
    m = _lr_penalised("L2", C, seed).fit(Xtr, ytr)
    return list(np.argsort(np.abs(m.coef_).mean(0))[::-1][:k]), m


def rf_importance(Xtr, ytr, k=None, trees=300, seed=42):
    rf = RandomForestClassifier(trees, class_weight="balanced_subsample", min_samples_leaf=3, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
    return (list(np.argsort(rf.feature_importances_)[::-1][:k]) if k else None), rf
