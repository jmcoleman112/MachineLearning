# random_forest.py
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def run_outer_cv_rf(X, y, n_trees, max_depth=10, random_state=42, n_features=None):
    """
    Ten-fold CV with wrapper-based feature selection.
    - If n_features is None => automatic stopping ("auto")
    - Else, force exactly n_features to be selected
    Returns:
        metrics: (10, 3) array of [acc, f1_macro, auc] per fold
        subset_sizes: (10,) array of number of features selected per fold
    """
    outer = KFold(n_splits=10, shuffle=True, random_state=random_state)
    metrics = []
    subset_sizes = []
    n_feat = "auto" if n_features is None else n_features

    for tr, te in outer.split(X, y):
        base_rf = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            random_state=random_state
        )

        sfs = SequentialFeatureSelector(
            estimator=base_rf,
            n_features_to_select=n_feat,
            direction="forward",
            cv=5
        )

        pipe = Pipeline([
            ("scaler", MinMaxScaler()),   # kept for structural parity with SVM script
            ("sfs", sfs),
            ("rf", base_rf)
        ])

        pipe.fit(X[tr], y[tr])
        yp = pipe.predict(X[te])
        proba = pipe.predict_proba(X[te])[:, 1]

        acc = accuracy_score(y[te], yp)
        f1m = f1_score(y[te], yp, average="macro")
        auc = roc_auc_score(y[te], proba)

        ksel = pipe.named_steps["sfs"].get_support(indices=True).size
        subset_sizes.append(ksel)
        metrics.append([acc, f1m, auc])

    return np.array(metrics), np.array(subset_sizes)


def print_rf(scores, n_trees, subset_sizes, type_label):
    mean, std = np.nanmean(scores, axis=0), np.nanstd(scores, axis=0)
    print(f"RandomForest (trees={n_trees}, type={type_label})")
    print(f"Mean acc={mean[0]:.4f}, F1={mean[1]:.4f}, AUC={mean[2]:.4f}")
    print(f"Std  acc={std[0]:.4f}, F1={std[1]:.4f}, AUC={std[2]:.4f}")
    print(f"Avg subset size={np.mean(subset_sizes):.2f}, per fold={subset_sizes}")
    print("-" * 40)


def run_all_rf_tests(X, y, random_state=42):
    """
    Part (i): 100 trees, max_depth=10, automatic SFS stop.
    Part (ii): 20 trees, forced SFS size = max selected in part (i).
    Prints results using print_rf and returns a dict of raw metrics.
    """
    results = {}

    # --- Part (i): 100 trees, automatic feature count ---
    metrics_100_auto, subset_sizes_100 = run_outer_cv_rf(
        X, y, n_trees=100, max_depth=10, random_state=random_state, n_features=None
    )
    print_rf(metrics_100_auto, 100, subset_sizes_100, "auto")
    results[("100", "auto")] = {
        "metrics": metrics_100_auto,
        "subset_sizes": subset_sizes_100
    }

    # Force feature count to highest selected across folds in part (i)
    forced_k = int(np.max(subset_sizes_100))

    # --- Part (ii): 20 trees, forced higher feature count ---
    metrics_20_fixed, subset_sizes_20_fixed = run_outer_cv_rf(
        X, y, n_trees=20, max_depth=10, random_state=random_state, n_features=forced_k
    )
    print_rf(metrics_20_fixed, 20, subset_sizes_20_fixed, f"fixed@{forced_k}")
    results[("20", f"fixed@{forced_k}")] = {
        "metrics": metrics_20_fixed,
        "subset_sizes": subset_sizes_20_fixed,
        "forced_features": forced_k
    }

    return results


if __name__ == "__main__":
    # Example usage (replace with real X, y)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 30))
    y = rng.integers(0, 2, size=120)

    _ = run_all_rf_tests(X, y)
