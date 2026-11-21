"""Metrics, subset evaluation, grouped cross-validation and the over/underfitting table."""
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.svm import SVC
from .config import Config
from .models import selection_classifier, make_classifiers


def evaluate_subset(Xtr, ytr, Xte, yte, cols, name, cfg: Config) -> dict:
    """Train the selection classifier on a feature subset; report train and test scores."""
    rf = selection_classifier(cfg).fit(Xtr[:, cols], ytr)
    ptr, pte = rf.predict(Xtr[:, cols]), rf.predict(Xte[:, cols])
    return {"Method": name, "# features": len(cols), "Train macro-F1": f1_score(ytr, ptr, average="macro"),
            "Test macro-F1": f1_score(yte, pte, average="macro"), "Test acc": accuracy_score(yte, pte)}


def test_metrics(yte, pte, proba=None) -> dict:
    """Accuracy, error rate, macro precision/recall/F1 and one-vs-rest macro ROC-AUC."""
    d = {"Accuracy": accuracy_score(yte, pte), "Error rate": 1 - accuracy_score(yte, pte),
         "Precision (macro)": precision_score(yte, pte, average="macro", zero_division=0),
         "Recall (macro)": recall_score(yte, pte, average="macro", zero_division=0),
         "F1 (macro)": f1_score(yte, pte, average="macro")}
    if proba is not None and proba.shape[1] == len(np.unique(yte)):
        d["ROC-AUC (OvR)"] = roc_auc_score(yte, proba, multi_class="ovr", average="macro")
    return d


def evaluate_models(Xtr, ytr, Xte, yte, feature_sets: dict, cfg: Config):
    """Train every classifier on every feature set and score it on the test set. Returns (table, fitted models)."""
    rows, fitted = [], {}
    for fs, cols in feature_sets.items():
        for name, mk in make_classifiers(cfg).items():
            m = mk().fit(Xtr[:, cols], ytr)
            row = {"Features": fs, "Model": name, **test_metrics(yte, m.predict(Xte[:, cols]), m.predict_proba(Xte[:, cols])),
                   "Train F1 (macro)": f1_score(ytr, m.predict(Xtr[:, cols]), average="macro")}
            rows.append(row); fitted[(fs, name)] = m; print(fs, "|", name, "done")
    return pd.DataFrame(rows), fitted


def cross_validate_models(Xtr, ytr, gtr, feature_sets: dict, cfg: Config) -> pd.DataFrame:
    """Lesion-grouped, stratified k-fold CV: a lesion never spans two folds."""
    cv = StratifiedGroupKFold(cfg.cv_folds, shuffle=True, random_state=cfg.seed); rows = []
    for fs, cols in feature_sets.items():
        for name, mk in make_classifiers(cfg).items():
            m = mk()
            if isinstance(m, SVC): m.set_params(probability=False)      # not needed for CV and much faster
            s = cross_validate(m, Xtr[:, cols], ytr, groups=gtr, cv=cv, scoring={"acc": "accuracy", "f1": "f1_macro"}, return_train_score=True)
            rows.append({"Features": fs, "Model": name, "CV acc": s["test_acc"].mean(), "CV acc std": s["test_acc"].std(),
                         "CV F1": s["test_f1"].mean(), "CV F1 std": s["test_f1"].std(), "CV train F1": s["train_f1"].mean()})
            print(fs, "|", name, "CV done")
    return pd.DataFrame(rows)


def best_by_cv(res: pd.DataFrame):
    """The model is chosen by cross-validated macro-F1, never by the test score."""
    return res.sort_values("CV F1", ascending=False).iloc[0]


def overfitting_table(res: pd.DataFrame, features: str) -> pd.DataFrame:
    sub = res[res["Features"] == features].set_index("Model")
    tab = pd.DataFrame({"Train F1": sub["Train F1 (macro)"], "CV F1": sub["CV F1"], "Test F1": sub["F1 (macro)"]})
    tab["Train-Test gap"] = tab["Train F1"] - tab["Test F1"]; return tab


def class_report(y, pred, classes) -> str:
    return classification_report(y, pred, target_names=classes, zero_division=0)
