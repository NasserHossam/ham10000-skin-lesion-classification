"""The classifiers used in the project."""
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from .config import Config


def selection_classifier(cfg: Config, seed=42):
    """Random Forest used to score every feature subset in the selection comparison."""
    return RandomForestClassifier(cfg.rf_trees, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)


def make_classifiers(cfg: Config, seed=42) -> dict:
    """Name -> factory for the five classifiers of the final comparison (all class-weighted except KNN)."""
    return {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
        "KNN (k=15)": lambda: KNeighborsClassifier(15, weights="distance"),
        "SVM (RBF)": lambda: SVC(C=1, class_weight="balanced", probability=True, random_state=seed),
        "Random Forest": lambda: RandomForestClassifier(cfg.rf_trees, min_samples_leaf=5, max_depth=12,
                                                        class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(max_iter=60 if cfg.fast else 200, learning_rate=0.05, max_depth=6,
                                                                       l2_regularization=1.0, class_weight="balanced", random_state=seed),
    }
