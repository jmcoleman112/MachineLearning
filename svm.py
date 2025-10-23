import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC

def run_outer_cv_svm(X, y, C_value, random_state=42, n_features = None):
    outer = KFold(n_splits=10, shuffle=True, random_state=random_state)
    metrics = []
    subset_sizes = []
    n_features = "auto" if n_features is None else n_features

    for tr, te in outer.split(X, y):
        base_svm = SVC(C=C_value, kernel="linear", probability=True)
        sfs = SequentialFeatureSelector(estimator=base_svm, n_features_to_select=n_features, direction="forward", cv=5)

        pipe = Pipeline([
            ("scaler", MinMaxScaler()),
            ("sfs", sfs),
            ("svm", base_svm)
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

def print_svm(scores, C, subset_sizes, type):
    mean, std = np.nanmean(scores, axis=0), np.nanstd(scores, axis=0)
    print(f"SVM (C={C}, type = {type})")
    print(f"Mean acc={mean[0]:.4f}, F1={mean[1]:.4f}, AUC={mean[2]:.4f}")
    print(f"Std  acc={std[0]:.4f}, F1={std[1]:.4f}, AUC={std[2]:.4f}")
    print(f"Avg subset size={np.mean(subset_sizes):.2f}, per fold={subset_sizes}")
    print("-" * 40)


