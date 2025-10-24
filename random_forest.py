import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

def run_outer_cv_rf(X, y, n_trees, n_features=None):
    outer = StratifiedKFold(n_splits=10, shuffle=True)
    metrics = []
    subset_sizes = []
    n_features = "auto" if n_features is None else n_features
    base_rf = RandomForestClassifier(n_estimators=n_trees, max_depth=10)
    sfs = SequentialFeatureSelector(estimator=clone(base_rf), n_features_to_select=n_features, tol=1e-4)
    for tr, te in outer.split(X, y):

        pipe = Pipeline([
            ("sfs", clone(sfs)),
            ("rf", clone(base_rf))
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

def print_rf(scores, n_trees, subset_sizes):
    mean, std = np.nanmean(scores, axis=0), np.nanstd(scores, axis=0)
    mean_subset = np.mean(subset_sizes)
    row = (
        f"{n_trees} & "
        f"{mean_subset:.0f} & "
        f"{mean[0]:.2f} & {std[0]:.2f} & "
        f"{mean[1]:.2f} & {std[1]:.2f} & "
        f"{mean[2]:.2f} & {std[2]:.2f}\\\\"
    )
    print(row)

