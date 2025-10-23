import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def run_outer_cv(X, y, K, weights="uniform", random_state=42):
    outer = KFold(n_splits=10, shuffle=True, random_state=random_state)

    base_knn = KNeighborsClassifier(n_neighbors=K, weights=weights)
    sfs = SequentialFeatureSelector(estimator=base_knn, n_features_to_select="auto", cv=10)
    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("sfs",sfs),
        ("knn", base_knn)
    ])

    metrics = []
    subset_sizes = []

    for tr, te in outer.split(X, y):
        pipe.fit(X[tr], y[tr])
        yp = pipe.predict(X[te])
        acc = accuracy_score(y[te], yp)
        f1m = f1_score(y[te], yp, average="macro")

        auc = roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1])

        metrics.append([acc, f1m, auc])

    return np.array(metrics)


def print_knn(scores, K ,weights):
    mean, std = np.nanmean(scores, axis=0), np.nanstd(scores, axis=0)
    print(f"K={K} ({weights})")
    print(f"Mean acc={mean[0]:.4f}, F1={mean[1]:.4f}, AUC={mean[2]:.4f}")
    print(f"Std  acc={std[0]:.4f}, F1={std[1]:.4f}, AUC={std[2]:.4f}")
    print("-" * 40)