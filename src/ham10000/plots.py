"""Figures used in the project (saved as PNG; no display needed)."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig); print("saved", path)


def scatter2d(Z, labels, title, ax, names=None):
    for k in np.unique(labels):
        m = labels == k
        ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=.6, label=names[k] if names is not None else k)
    ax.set_title(title)


def eda_figures(raw, imgs, meta, classes, fig_dir):
    order = raw["dx"].value_counts().index
    fig, ax = plt.subplots(figsize=(7, 4)); sns.countplot(data=raw, x="dx", order=order, ax=ax)
    for p in ax.patches:
        ax.annotate(f"{100 * p.get_height() / len(raw):.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=9)
    ax.set_title("Class distribution"); _save(fig, fig_dir / "eda_class_dist.png")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    sns.boxplot(data=raw, x="dx", y="age", order=order, ax=ax[0]); ax[0].set_title("Age by diagnosis")
    sns.countplot(data=raw, x="dx", hue="sex", order=order, ax=ax[1]); ax[1].set_title("Sex by diagnosis"); _save(fig, fig_dir / "eda_age_sex.png")
    fig = plt.figure(figsize=(11, 5)); sns.countplot(data=raw, y="localization", hue="dx", hue_order=order)
    plt.title("Lesion location by diagnosis"); _save(fig, fig_dir / "eda_location.png")
    rng = np.random.default_rng(0); fig, axes = plt.subplots(len(classes), 5, figsize=(10, 2 * len(classes)))
    for r, c in enumerate(classes):
        pool = np.where(meta["dx"].values == c)[0]; ids = rng.choice(pool, min(5, len(pool)), replace=False)
        for k in range(5):
            axes[r, k].axis("off")
            if k < len(ids): axes[r, k].imshow(imgs[ids[k]])
        axes[r, 0].set_title(c, loc="left", fontsize=10)
    _save(fig, fig_dir / "eda_samples.png")


def linear_figures(info, X_raw_tr, ytr, classes, fig_dir, km=None, ztr2=None):
    rng = np.random.default_rng(0); sub = rng.choice(len(ytr), min(3000, len(ytr)), replace=False)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ax[0].plot(info["cum"]); ax[0].axhline(.95, ls="--", c="r"); ax[0].set_title("PCA cumulative explained variance"); ax[0].set_xlabel("components")
    top2 = np.argsort(X_raw_tr.var(0))[-2:]
    scatter2d(X_raw_tr[sub][:, top2], ytr[sub], "Before reduction (2 raw pixels)", ax[1])
    scatter2d(info["pca"].transform(info["Xtr"])[:, :2][sub], ytr[sub], "After PCA (PC1 vs PC2)", ax[2], classes); ax[2].legend(markerscale=3, fontsize=8)
    _save(fig, fig_dir / "pca.png")
    lda_tr = info["lda"].transform(info["Xtr"]); svd_tr = info["svd"].transform(X_raw_tr)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    scatter2d(lda_tr[sub], ytr[sub], "After LDA (LD1 vs LD2)", ax[0], classes); ax[0].legend(markerscale=3, fontsize=8)
    scatter2d(svd_tr[sub], ytr[sub], "After SVD (comp 1 vs 2)", ax[1]); _save(fig, fig_dir / "lda_svd.png")
    if km is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5)); Z2 = info["Ztr50"][:, :2]
        scatter2d(Z2[sub], ytr[sub], "True classes (PCA space)", ax[0], classes); ax[0].legend(markerscale=3, fontsize=8)
        scatter2d(Z2[sub], km.labels_[sub], "K-Means clusters (k=7)", ax[1]); ax[1].legend(markerscale=3, fontsize=8)
        _save(fig, fig_dir / "kmeans.png")


def nonlinear_figure(emb, yvis, classes, fig_dir):
    order = [m for m in ["Kernel PCA (RBF)", "t-SNE", "SNE", "UMAP", "Metric MDS", "Non-metric MDS", "Isomap", "LLE"] if m in emb]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, name in zip(axes.ravel(), order): scatter2d(emb[name], yvis, name, ax, classes)
    for ax in axes.ravel()[len(order):]: ax.axis("off")
    axes[0, 0].legend(markerscale=3, fontsize=8); _save(fig, fig_dir / "nonlinear_all.png")


def filter_figure(mi, fs, names, fig_dir):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for a, sc, t in zip(ax, [mi, fs], ["Mutual information", "ANOVA F-score"]):
        o = np.argsort(sc)[::-1][:20]; a.barh([names[i] for i in o][::-1], sc[o][::-1]); a.set_title(f"Top 20 features: {t}")
    _save(fig, fig_dir / "filter_scores.png")


def embedded_figure(items, names, fig_dir):
    fig, ax = plt.subplots(1, len(items), figsize=(6 * len(items), 6))
    for a, (t, v) in zip(np.atleast_1d(ax), items):
        o = np.argsort(v)[::-1][:15]; a.barh([names[i] for i in o][::-1], v[o][::-1]); a.set_title(t)
    _save(fig, fig_dir / "embedded.png")


def ga_figure(best_hist, mean_hist, fig_dir):
    fig = plt.figure(figsize=(6, 4)); plt.plot(best_hist, label="best")
    if mean_hist is not None: plt.plot(mean_hist, label="mean")
    plt.xlabel("generation"); plt.ylabel("fitness"); plt.title("GA convergence"); plt.legend(); _save(fig, fig_dir / "ga_convergence.png")


def selection_summary_figure(res_sel, fig_dir):
    fig = plt.figure(figsize=(8, 5)); plt.scatter(res_sel["# features"], res_sel["Test macro-F1"])
    for _, r in res_sel.iterrows(): plt.annotate(r["Method"].split(":")[0], (r["# features"], r["Test macro-F1"]), fontsize=7)
    plt.xlabel("# features"); plt.ylabel("Test macro-F1"); plt.title("Feature selection: size vs performance"); _save(fig, fig_dir / "selection_summary.png")


def confusion_roc_figure(y, pred, proba, classes, fig_dir):
    fig, ax = plt.subplots(1, 3, figsize=(21, 6))
    ConfusionMatrixDisplay(confusion_matrix(y, pred), display_labels=classes).plot(ax=ax[0], colorbar=False); ax[0].set_title("Confusion matrix (counts)")
    ConfusionMatrixDisplay(confusion_matrix(y, pred, normalize="true"), display_labels=classes).plot(ax=ax[1], colorbar=False, values_format=".2f")
    ax[1].set_title("Confusion matrix (row-normalised = recall)")
    Y = label_binarize(y, classes=np.arange(len(classes)))
    for k, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(Y[:, k], proba[:, k]); ax[2].plot(fpr, tpr, label=f"{c} (AUC {auc(fpr, tpr):.2f})")
    ax[2].plot([0, 1], [0, 1], "k--", lw=1); ax[2].set_xlabel("False positive rate"); ax[2].set_ylabel("True positive rate")
    ax[2].set_title("ROC curves (one-vs-rest)"); ax[2].legend(fontsize=8); _save(fig, fig_dir / "final_eval.png")


def overfitting_figure(tab, title, fig_dir):
    ax = tab[["Train F1", "CV F1", "Test F1"]].plot.bar(figsize=(9, 4.5), rot=20)
    ax.set_title(title); ax.set_ylabel("macro-F1"); _save(ax.get_figure(), fig_dir / "overfitting.png")


def bonus_figures(P, S, hists, masks, seeds, names, fig_dir, order):
    """Boxplot, size/score trade-off, metaheuristic convergence, Pareto fronts and selection-frequency heatmap."""
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(7.2, 3.4)); ax.boxplot([P[m].values for m in order], showmeans=True)
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels([m.replace(": ", ":\n") for m in order], fontsize=6.5)
    ax.set_ylabel("test macro-F1"); ax.set_title(f"Repeated lesion-level 80/20 splits (n = {len(seeds)} seeds)"); _save(fig, fig_dir / "multiseed_box.png")
    fig, ax = plt.subplots(figsize=(3.4, 3.0)); ax.errorbar(S["n"], S["f1"], yerr=S["f1s"].fillna(0), fmt="o", capsize=2)
    for m, r in S.iterrows(): ax.annotate(m.split(":")[-1].strip(), (r["n"], r["f1"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean # features"); ax.set_ylabel("test macro-F1"); _save(fig, fig_dir / "tradeoff.png")
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for name in ["GA", "BPSO", "NSGA-II"]:
        H = np.array([hists[str(s)][name] for s in seeds if str(s) in hists]); ax.plot(H.mean(0), label=name)
        ax.fill_between(range(H.shape[1]), H.mean(0) - H.std(0), H.mean(0) + H.std(0), alpha=0.15)
    ax.set_xlabel("generation / iteration"); ax.set_ylabel("best inner-validation fitness"); ax.legend(fontsize=7); _save(fig, fig_dir / "convergence_meta.png")
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for s in seeds:
        pf = hists[str(s)]["pareto"]; o = np.argsort(pf["size"]); ax.plot(np.array(pf["size"])[o], np.array(pf["f1"])[o], "o-", ms=3, alpha=0.6)
    ax.set_xlabel("# selected features"); ax.set_ylabel("inner-validation macro-F1"); ax.set_title("NSGA-II Pareto fronts", fontsize=8); _save(fig, fig_dir / "pareto.png")
    cols = ["Filter: MI", "Wrapper: backward", "Embedded: L1", "Embedded: RF importance", "GA", "BPSO", "NSGA-II (knee)"]
    nf = len(names); freq = pd.DataFrame({m: np.bincount(np.concatenate([masks[str(s)][m] for s in seeds if str(s) in masks]), minlength=nf) / len(seeds) for m in cols}, index=names)
    freq = freq.loc[freq.mean(1).sort_values(ascending=False).index[:25]]
    fig, ax = plt.subplots(figsize=(3.6, 5.2)); im = ax.imshow(freq.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(freq))); ax.set_yticklabels(freq.index, fontsize=6); ax.set_xticks(range(freq.shape[1]))
    ax.set_xticklabels([c.replace(": ", ":\n") for c in freq.columns], fontsize=5.5, rotation=90); plt.colorbar(im, ax=ax, label="selection frequency")
    _save(fig, fig_dir / "stability_heatmap.png")
