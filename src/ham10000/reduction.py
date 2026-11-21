"""Dimensionality reduction (linear and non-linear) and clustering diagnostics (Phase 1, Steps 2 and 3)."""
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding, trustworthiness
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def scale_on_train(X, is_train):
    """Standardise with statistics from the training rows only."""
    sc = StandardScaler().fit(X[is_train]); return sc.transform(X), sc


def _lr(A, y):
    return LogisticRegression(max_iter=3000, class_weight="balanced").fit(A, y)


def linear_reduction(X_raw, y, is_train):
    """PCA, LDA and SVD on the 784 grayscale pixels; compare them with one classifier. Returns (table, info)."""
    Xs, _ = scale_on_train(X_raw, is_train)
    Xtr, Xte, ytr, yte = Xs[is_train], Xs[~is_train], y[is_train], y[~is_train]
    pca = PCA().fit(Xtr); cum = np.cumsum(pca.explained_variance_ratio_)
    n90, n95 = int(np.searchsorted(cum, .90) + 1), int(np.searchsorted(cum, .95) + 1)
    n_comp = min(50, Xtr.shape[0] - 1, Xtr.shape[1])
    pca50 = PCA(n_comp, random_state=42).fit(Xtr)
    lda = LDA(solver="svd").fit(Xtr, ytr)
    svd = TruncatedSVD(n_comp, random_state=42).fit(X_raw[is_train])
    reps = {"Raw pixels (784)": (Xtr, Xte), f"PCA ({n_comp})": (pca50.transform(Xtr), pca50.transform(Xte)),
            f"LDA ({lda.transform(Xtr[:2]).shape[1]})": (lda.transform(Xtr), lda.transform(Xte)),
            f"SVD ({n_comp})": (svd.transform(X_raw[is_train]), svd.transform(X_raw[~is_train]))}
    rows = []
    for name, (A, B) in reps.items():
        clf = _lr(A, ytr); ptr, pte = clf.predict(A), clf.predict(B)
        rows.append({"Representation": name, "Train acc": accuracy_score(ytr, ptr), "Test acc": accuracy_score(yte, pte),
                     "Train macro-F1": f1_score(ytr, ptr, average="macro"), "Test macro-F1": f1_score(yte, pte, average="macro")})
    info = dict(Xs=Xs, pca=pca, pca50=pca50, lda=lda, svd=svd, cum=cum, n90=n90, n95=n95,
                Ztr50=pca50.transform(Xtr), Zte50=pca50.transform(Xte), ytr=ytr, yte=yte, Xtr=Xtr)
    return pd.DataFrame(rows).round(3), info


def kmeans_report(Z, y, k=7, seed=42, sub=3000):
    """K-Means on a reduced space compared with the true classes."""
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    idx = np.random.default_rng(0).choice(len(Z), min(sub, len(Z)), replace=False)
    return km, {"ARI": adjusted_rand_score(y, km.labels_), "NMI": normalized_mutual_info_score(y, km.labels_),
                "Silhouette": silhouette_score(Z[idx], km.labels_[idx])}


# ---------------- vanilla SNE (scikit-learn has none) ----------------
def cond_p(X, perplexity=30, iters=60):
    """Conditional affinities p(j|i) with a bandwidth found by bisection to match the perplexity."""
    D = squareform(pdist(X, "sqeuclidean")); D = D / np.median(D)
    n = len(X); beta = np.ones(n); lo = np.zeros(n); hi = np.full(n, np.inf); target = np.log(perplexity)
    for _ in range(iters):
        P = np.exp(-D * beta[:, None]); np.fill_diagonal(P, 0); P /= P.sum(1, keepdims=True) + 1e-12
        H = -(P * np.log(P + 1e-12)).sum(1); high = H > target
        lo = np.where(high, beta, lo); hi = np.where(high, hi, beta)
        beta = np.where(high, np.where(np.isinf(hi), beta * 2, (beta + hi) / 2), (beta + lo) / 2)
    return P


def sne(X, perplexity=30, n_iter=500, lr=0.2, seed=0):
    """Vanilla SNE (Hinton and Roweis, 2002): Gaussian affinities in both spaces, asymmetric KL divergence."""
    P = cond_p(X, perplexity); r = np.random.default_rng(seed)
    Y = r.normal(0, 1e-2, (len(X), 2)); V = np.zeros_like(Y)
    for it in range(n_iter):
        Q = np.exp(-squareform(pdist(Y, "sqeuclidean"))); np.fill_diagonal(Q, 0); Q /= Q.sum(1, keepdims=True) + 1e-12
        M = P - Q + P.T - Q.T
        grad = 2 * (M.sum(1)[:, None] * Y - M @ Y)
        V = (0.5 if it < 100 else 0.8) * V - lr * grad; Y = Y + V
    return Y


def nonlinear_reduction(Ztr, Zte, ytr, yte, fit_n=3000, vis_n=1500, seed=1):
    """Kernel PCA, Isomap, LLE, UMAP (can embed new data) and t-SNE, SNE, MDS (visualisation only).

    Everything works on the 50-D PCA space. Returns (results table, 2-D embeddings of the plotted points, their labels).
    """
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(len(Ztr), min(fit_n, len(Ztr)), replace=False)
    vis = fit_idx[:min(vis_n, len(fit_idx))]
    Zfit, Zvis, yvis = Ztr[fit_idx], Ztr[vis], ytr[vis]
    gamma = 1.0 / Ztr.var(0).sum()
    makers = {"Kernel PCA (RBF)": lambda d: KernelPCA(d, kernel="rbf", gamma=gamma),
              "Isomap": lambda d: Isomap(n_neighbors=15, n_components=d),
              "LLE": lambda d: LocallyLinearEmbedding(n_neighbors=30, n_components=d, reg=1e-1, random_state=42)}
    try:
        import umap
        makers["UMAP"] = lambda d: umap.UMAP(n_components=d, n_neighbors=15, random_state=42)
    except ImportError:
        print("umap-learn is not installed: UMAP is skipped (pip install umap-learn)")
    emb, rows = {}, []
    for name, mk in makers.items():
        emb[name] = mk(2).fit(Zfit).transform(Zvis)
        r10 = mk(10).fit(Zfit); Atr, Ate = r10.transform(Ztr), r10.transform(Zte)
        clf = _lr(Atr, ytr)
        rows.append({"Method": name, "Trustworthiness": trustworthiness(Zvis, emb[name], n_neighbors=10),
                     "Train macro-F1": f1_score(ytr, clf.predict(Atr), average="macro"),
                     "Test macro-F1": f1_score(yte, clf.predict(Ate), average="macro"),
                     "Test acc": accuracy_score(yte, clf.predict(Ate))})
        print(name, "done")
    emb["t-SNE"] = TSNE(2, perplexity=min(30, len(Zvis) - 1), init="pca", random_state=42).fit_transform(Zvis)
    emb["SNE"] = sne(Zvis, perplexity=min(30, len(Zvis) // 4))
    emb["Metric MDS"] = MDS(2, metric=True, n_init=1, max_iter=100, random_state=42).fit_transform(Zvis)
    # non-metric MDS is started from the metric solution (a random start collapsed in our first run)
    emb["Non-metric MDS"] = MDS(2, metric=False, n_init=1, max_iter=300, random_state=42).fit_transform(Zvis, init=emb["Metric MDS"])
    for n in ["t-SNE", "SNE", "Metric MDS", "Non-metric MDS"]:
        rows.append({"Method": n, "Trustworthiness": trustworthiness(Zvis, emb[n], n_neighbors=10)})
    return pd.DataFrame(rows).round(3), emb, yvis
