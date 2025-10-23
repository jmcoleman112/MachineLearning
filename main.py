from Feature_Extraction import *
from knn import *
from svm import *

X, y = feature_extraction()

# KNN
results = {}
for K in [1, 3, 5, 7, 9]:
    scores = run_outer_cv(X, y, K=K, weights="uniform")
    print_knn(scores, K, "uniform")
    results[K] = scores

best_K = max(results, key=lambda k: np.nanmean(results[k][:,1]))

scores = run_outer_cv(X, y, K=best_K, weights="distance")
print_knn(scores, best_K, "distance")

# SVM
for C in [0.1, 1, 10]:
    print(f"\n--- SVM with C = {C} ---")

    metrics_auto, subset_sizes = run_outer_cv_svm(X, y, C)
    print_svm(metrics_auto, C, subset_sizes, "auto")
    max_feats = np.max(subset_sizes)

    metrics_fixed, subset_sizes = run_outer_cv_svm(X, y, C, n_features=max_feats)
    print_svm(metrics_fixed, C, subset_sizes, "fixed")







