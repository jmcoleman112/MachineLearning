import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.base import clone

def run_outer_cv(X, y, K, weights="uniform"):
    outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=21207103)
    inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=21207103+1)
    base_knn = KNeighborsClassifier(n_neighbors=K, weights=weights)
    sfs = SequentialFeatureSelector(
        estimator=clone(base_knn),
        n_features_to_select="auto",
        direction="forward",
        scoring="roc_auc",
        tol=1e-5,
        cv=inner,
        n_jobs=-1
    )

    metrics = []

    for tr, te in outer.split(X, y):
        pipe = Pipeline([
            ("sfs", clone(sfs)),
            ("knn", clone(base_knn))
        ])
        pipe.fit(X[tr], y[tr])
        yp = pipe.predict(X[te])
        acc = accuracy_score(y[te], yp)
        f1m = f1_score(y[te], yp, average="macro")

        auc = roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1])

        metrics.append([acc, f1m, auc])

    return np.array(metrics)

def print_knn(scores, K, weights):
    mean, std = np.nanmean(scores, axis=0), np.nanstd(scores, axis=0)
    row = (
        f"{K} & "
        f"{mean[0]:.2f} & {std[0]:.2f} & "
        f"{mean[1]:.2f} & {std[1]:.2f} & "
        f"{mean[2]:.2f} & {std[2]:.2f}\\\\"
    )
    print(row)
