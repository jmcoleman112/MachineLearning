import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.base import clone


def run_outer_cv_svm(X, y, C_value, n_features=None):
    outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=21207103)
    inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=21207103+1)
    n_features = "auto" if n_features is None else n_features
    base_svm = SVC(C=C_value, kernel='rbf')
    sfs = SequentialFeatureSelector(
        estimator=clone(base_svm),
        n_features_to_select=n_features,
        direction="forward",
        scoring="roc_auc",
        tol=1e-5,
        cv=inner,
        n_jobs=-1
    )
    metrics = []
    n_selected = []

    for tr, te in outer.split(X, y):
        pipe = Pipeline([
            ("sfs", clone(sfs)),
            ("svm", clone(base_svm))
        ])

        pipe.fit(X[tr], y[tr])
        n_selected.append(len(pipe.named_steps["sfs"].get_support(indices=True)))

        yp = pipe.predict(X[te])
        scores = pipe.decision_function(X[te])
        acc = accuracy_score(y[te], yp)
        f1 = f1_score(y[te], yp, average="binary")
        auc = roc_auc_score(y[te], scores)
        metrics.append([acc, f1, auc])

    return np.array(metrics), np.array(n_selected)

def print_svm(scores, C, subset_sizes):
    mean, std = np.nanmean(scores, axis=0), np.nanstd(scores, axis=0)
    mean_subset = np.mean(subset_sizes)

    # Header (prints once for reference)

    # Row formatted for LaTeX
    row = (
        f"{C} & "
        f"{mean_subset:.0f} & "
        f"{mean[0]:.2f} & {std[0]:.2f} & "
        f"{mean[1]:.2f} & {std[1]:.2f} & "
        f"{mean[2]:.2f} & {std[2]:.2f}\\\\"
    )
    print(row)




