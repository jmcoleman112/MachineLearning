from Feature_Extraction import *
from print_figures import *
from knn import *
from svm import *
from random_forest import *

X, y, feature_names = feature_extraction()
# print_feature_stats(X, feature_names)
# analyze_feature_distributions(X, y, feature_names)
X= minmax_scale_features(X)
print_feature_stats(X, feature_names)

input = [1,1,1]
if input[0]==1:
    results = {}
    print("K  & Mean Acc. & Acc. Std. Dev. & Mean F1 & F1 Std. Dev. & Mean AUC & AUC Std. Dev")
    for K in [1, 3, 5, 7, 9]:
        scores = run_outer_cv(X, y, K=K, weights="uniform")
        print_knn(scores, K, "uniform")
        results[K] = scores

    results = {}
    print("K  & Mean Acc. & Acc. Std. Dev. & Mean F1 & F1 Std. Dev. & Mean AUC & AUC Std. Dev")
    for K in [1, 3, 5, 7, 9]:
        scores = run_outer_cv(X, y, K=K, weights="distance")
        print_knn(scores, K, "uniform")
        results[K] = scores

# SVM
if input[1]==1:
    print("SVM Auto")
    print("C  & Mean Feature Ct. & Mean Acc. & Acc. SD & Mean F1 & F1 SD & Mean AUC & AUC SD\\\\")
    for C in [0.1, 1, 10]:

        metrics_auto, subset_sizes = run_outer_cv_svm(X, y, C)
        print_svm(metrics_auto, C, subset_sizes)
        max_feats = np.max(subset_sizes)

    print("SVM Fixed")
    print("C  & Mean Feature Ct. & Mean Acc. & Acc. SD & Mean F1 & F1 SD & Mean AUC & AUC SD\\\\")
    for C in [0.1, 1, 10]:
        metrics_fixed, subset_sizes = run_outer_cv_svm(X, y, C, n_features=max_feats)
        print_svm(metrics_fixed, C, subset_sizes)

if input[2]==1:
    print("Random Forest")
    print("Trees  & Mean Feature Ct. & Mean Acc. & Acc. SD & Mean F1 & F1 SD & Mean AUC & AUC SD\\\\")

    for n_trees in [100, 20]:
        # --- 1) AUTO run ---
        scores_auto, sizes_auto = run_outer_cv_rf(X, y, n_trees, n_features=None)  # None -> "auto" inside your fn
        print_rf(scores_auto, n_trees, sizes_auto)

        # figure out max selected under AUTO
        k_max = int(np.max(sizes_auto))
        k_plus = min(k_max + 1, X.shape[1])

        # --- 2) (k_max + 1) run ---
        scores_kplus, sizes_kplus = run_outer_cv_rf(X, y, n_trees, n_features=k_plus)
        print_rf(scores_kplus, n_trees, sizes_kplus)







