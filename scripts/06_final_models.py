import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ham10000.config import add_common_args, config_from_args, CLASSES
import json
import pandas as pd
from ham10000.evaluation import evaluate_models, cross_validate_models, best_by_cv, overfitting_table, class_report
from ham10000.pipeline import handcrafted_data, save_table
from ham10000.plots import confusion_roc_figure, overfitting_figure


def main(argv=None):
    """Step 6: five classifiers on three feature sets, 80/20 test metrics, grouped cross-validation, confusion matrix, ROC and overfitting."""
    args = add_common_args(argparse.ArgumentParser(description=main.__doc__)).parse_args(argv); cfg = config_from_args(args)
    d = handcrafted_data(cfg); names, Xs, y, is_tr, groups = d["names"], d["Xs"], d["y"], d["is_train"], d["groups"]
    Xtr, Xte, ytr, yte, gtr = Xs[is_tr], Xs[~is_tr], y[is_tr], y[~is_tr], groups[is_tr]
    sel = json.loads((cfg.results_dir / "selected_features.json").read_text()); idx = {n: i for i, n in enumerate(names)}
    sets = {f"All ({len(names)})": list(range(len(names)))}
    for key in ("GA", "L1"): sets[f"{key} ({len(sel[key])})"] = [idx[n] for n in sel[key]]
    res80, fitted = evaluate_models(Xtr, ytr, Xte, yte, sets, cfg)
    cv = cross_validate_models(Xtr, ytr, gtr, sets, cfg)
    res = res80.merge(cv, on=["Features", "Model"]).round(3); save_table(res, cfg.results_dir / "res_final_models.csv")
    show = res.sort_values("CV F1", ascending=False)
    print(show[["Features", "Model", "CV acc", "CV acc std", "CV F1", "CV F1 std"]].to_string(index=False)); print()
    print(show[["Features", "Model", "Accuracy", "Error rate", "Precision (macro)", "Recall (macro)", "F1 (macro)", "ROC-AUC (OvR)"]].to_string(index=False))
    best = best_by_cv(res); print("\nBest by cross-validation:", best["Features"], "+", best["Model"])
    cols = sets[best["Features"]]; model = fitted[(best["Features"], best["Model"])]
    pte, proba = model.predict(Xte[:, cols]), model.predict_proba(Xte[:, cols]); print(class_report(yte, pte, CLASSES))
    confusion_roc_figure(yte, pte, proba, CLASSES, cfg.figures_dir)
    tab = overfitting_table(res, best["Features"]); print(tab.round(3).to_string()); overfitting_figure(tab, f"Train vs CV vs Test macro-F1 ({best['Features']})", cfg.figures_dir)


if __name__ == "__main__":
    main()
