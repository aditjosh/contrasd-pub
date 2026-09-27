import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import cdist

############ Calculating conformal set sizes for KNN and KMeans models
class CustomException(Exception):
    """Custom exception for invalid values."""
    def __init__(self, message="Found invalid value"):
        super().__init__(f"{message}")

def calculate_calibration_conformity_scores_loo(calib_embeddings, calib_labels):
        """
        Calculates the conformity score (alpha) for the calibration set using 
        Leave-One-Out (LOO) L2-Normalized Euclidean Distance.
        
        Alpha_i = min_j!=i, y_j=y_i { Euclidean Distance(emb_i, emb_j) }
        
        This prevents the distance being 0 (distance to self), which causes tiny q_hat.
        
        Returns: A dictionary mapping class IDs to a list of non-conformity scores.
        """
        unique_classes = np.unique(calib_labels)
        
        # Pre-calculate distances between ALL pairs in the calibration set
        # These embeddings for are NOT L2-normalized
        distances_matrix = cdist(calib_embeddings, calib_embeddings, metric='euclidean')
    
        scores = {c.item(): [] for c in unique_classes}
    
        for i in range(len(calib_embeddings)):
            current_label = calib_labels[i]
            
            # Mask for points of the SAME class, excluding the current point (i != j)
            mask = (calib_labels == current_label)
            
            # Create a boolean tensor mask from the numpy mask
            #mask_tensor = same_class_mask.detach().clone()
            
            # Ensure we exclude the point itself (i != i)
            mask[i] = False 
            
            if mask.sum() > 0:
                # Look at the distances from point 'i' to all other points
                row_i = distances_matrix[i]
                
                # Find the minimum distance to any OTHER point of the SAME class
                min_distance = row_i[mask].min()
                scores[current_label.item()].append(min_distance.item())
    
        return scores

def calculate_test_conformity_scores(calib_embeddings, calib_labels, test_embeddings):
    """
    Calculates the conformity score (alpha) for the test set.
    
    Alpha_i(c) = min_j, y_j=c { Euclidean Distance(emb_i, emb_j) }
    
    Returns: A list of dictionaries: [{class: score}, ...]
    """
    unique_classes = np.unique(calib_labels)
    
    all_test_scores = []

    for i in range(len(test_embeddings)):
        test_emb = np.expand_dims(test_embeddings[i],axis=0)
        
        # Calculate Euclidean distance between the test sample and ALL calibration samples
        # Shape: (1, num_calib_samples)
        distances = np.squeeze(cdist(test_emb, calib_embeddings, metric='euclidean'),axis=0)
        
        test_scores_i = {}

        for c in unique_classes:
            # Mask for calibration points belonging to class 'c'
            mask = (calib_labels == c)
            #mask = torch.tensor(same_class_mask.detach().clone(), dtype=torch.bool).to(self.device)
            
            if mask.sum() > 0:
                # Find the MINIMUM distance (most similar/closest point) in class 'c'
                min_distance = distances[mask].min()
                test_scores_i[c.item()] = min_distance.item()
            else:
                 # If no calibration points exist for class c, score is effectively infinite
                 test_scores_i[c.item()] = np.inf 

        all_test_scores.append(test_scores_i)

    return all_test_scores

def critical_quantile_mc(calib_alphas, confidence_level=0.9, n_bootstrap_samples=500):
    """
    Calculates Quantile q_hat with Monte Carlo (Bootstrap) Sampling
    """
    q_hat = {}
    gamma = 1 - confidence_level # Significance level
    N_BOOTSTRAP = n_bootstrap_samples # Number of Monte Carlo samples for robustness
    quantile_percentile = confidence_level * 100

    for c, alphas in calib_alphas.items():
        if not alphas: continue
        
        n = len(alphas)
        # Target index for the (1-gamma) quantile
        idx = int(np.ceil((n + 1) * gamma)) - 1
        idx = max(0, min(idx, n - 1))
        
        # Monte Carlo (Bootstrap) Loop to stabilize q_hat
        # We resample the calibration scores to ensure the chosen threshold 
        # is not overly sensitive to specific outliers in the calibration set.
        mc_quantiles = []
        alphas_array = np.array(alphas)
        
        for _ in range(N_BOOTSTRAP):
            # Sample with replacement
            resampled = np.random.choice(alphas_array, size=n, replace=True)
            resampled.sort()
            critical_quantile = np.percentile(resampled, quantile_percentile, axis=0)
            #mc_quantiles.append(resampled[idx])
            mc_quantiles.append(critical_quantile)
            
        # Use the mean of the MC quantiles for a robust estimate
        q_hat[c] = np.mean(mc_quantiles)
        
        #print(f"Class {c}: n={n}, MC Robust q_hat = {q_hat[c]:.4f} (std: {np.std(mc_quantiles):.4f})")

    return q_hat

def compute_prediction_set(test_scores, q_hat):
    """
    Computes the prediction set for a single test sample based on the quantile q_hat.
    
    The prediction set includes all classes 'c' where the test sample's non-conformity 
    score for that class is less than or equal to the quantile q_hat.
    """
    prediction_set = []
    
    # test_scores is a dictionary: {class: score_for_this_test_sample}
    # Example: {'0': 0.1, '1': 0.8}
    
    for c, alpha_score in test_scores.items():
        #print(alpha_score)
        #print(f"q_hat[c]: {q_hat[c]}")
        ## Scores are negatively-oriented (smaller scores -- distances -- are better) 
        if alpha_score <= q_hat[c]:
            prediction_set.append(c)
            
    return sorted(prediction_set)

def compute_conditional_coverage(q_hat, all_test_scores, test_labels):
    prediction_sets = []
    coverage_count = 0
    total_sets = len(test_labels)
    
    # Initialize structures for Conditional Coverage tracking
    unique_classes = np.unique(test_labels)
    conditional_coverage_total = {c: 0 for c in unique_classes}
    conditional_coverage_correct = {c: 0 for c in unique_classes}

    for i in range(total_sets):
        current_true_label = test_labels[i]
        scores_i = all_test_scores[i] 
        
        # Find the set of possible labels (C(x))
        C_x = compute_prediction_set(scores_i, q_hat)
        prediction_sets.append(C_x)
        
        # Update conditional coverage tracking
        conditional_coverage_total[current_true_label.item()] += 1
        
        if current_true_label in C_x:
            coverage_count += 1 # Marginal Coverage
            conditional_coverage_correct[current_true_label.item()] += 1

    return prediction_sets, coverage_count, conditional_coverage_total, conditional_coverage_correct


def conformal_prediction_kmeans_with_synthetic_data(kmeans_model, calib_data, val_data, clusters_labels_mapping, alpha=0.1, print_stats=False):
    calib_emb = calib_data[0]
    calib_labels = calib_data[1]
    val_emb = val_data[0]
    val_labels = val_data[1]
    
    len_calib_data = calib_emb.shape[0]
    len_val_data = val_emb.shape[0]

    ############################# Clustering model
    y_calib_pred = kmeans_model.predict(calib_emb)
    y_calib_pred_relab = np.vectorize(clusters_labels_mapping.get)(y_calib_pred)
    y_val_pred = kmeans_model.predict(val_emb)
    y_val_pred_relab = np.vectorize(clusters_labels_mapping.get)(y_val_pred)

    
    ############################### Conformal Calibration
    # Compute LOO Non-Conformity Scores (Alpha Values) for Calibration Set
    # This calculation ensures no 0 distance scores are included.
    calib_alphas = calculate_calibration_conformity_scores_loo(calib_emb, y_calib_pred_relab)

    # Calculate Quantile q_hat with Monte Carlo (Bootstrap) Sampling
    CONFIDENCE_LEVEL = 1 - alpha
    n_bootstrap_samples = 500
    q_hat = critical_quantile_mc(calib_alphas, confidence_level=CONFIDENCE_LEVEL, n_bootstrap_samples=n_bootstrap_samples)
    #list_q_hat.append(q_hat)
    
    if print_stats:
        print(f"q_hat values: {q_hat}")

    # Compute Test Set Non-Conformity Scores
    # Distance from test point to the nearest calibration point in each class.
    all_test_scores = calculate_test_conformity_scores(calib_emb, y_calib_pred_relab, val_emb)
    #print(all_test_scores)
    #print(test_labels)

    # 5. Determine Prediction Sets
    try:
        prediction_sets, coverage_count, conditional_coverage_total, conditional_coverage_correct = compute_conditional_coverage(q_hat, all_test_scores, y_val_pred_relab)
    except KeyError:
        raise CustomException('Key error in q_hat, continue to next iter to recompute')
        
    # --- 3.4. Evaluation ---
    total_sets = len(prediction_sets)
    coverage = coverage_count / total_sets
    avg_set_size = np.mean([len(s) for s in prediction_sets])
    set_sizes = [len(s) for s in prediction_sets]
    
    if print_stats:
        print("\n--- Conformal Prediction Results ---")
        print(f"Target Confidence (1-alpha): {CONFIDENCE_LEVEL:.2f}")
        print(f"Empirical Marginal Coverage: {coverage:.4f}")
        print(f"Average Prediction Set Size: {avg_set_size:.3f}")

    counts = {}
    for item in set_sizes:
        counts[item] = counts.get(item, 0) + 1
    #list_set_sizes_per_class.append(counts)
    if print_stats:
        print(f"Prediction Set Sizes Per Class: {counts}")
        print(f"Proportion of Set Size 1: {counts[1]/sum(counts.values())}")
    
    print("\n--- Empirical Conditional Coverage ---")
    # This shows the actual coverage achieved for each specific class
    
    cc_per_class = {}
    for c in sorted(conditional_coverage_total.keys()):
        total = conditional_coverage_total[c]
        correct = conditional_coverage_correct[c]
        if total > 0:
            cond_cov = correct / total
            cc_per_class[c] = cond_cov
            if print_stats:
                print(f"  Class {c} Coverage (n={total}): {cond_cov:.4f}")
                
        else:
            if print_stats:
                print(f"  Class {c} Coverage (n=0): N/A (No test samples for this class)")
    #list_cc_per_class.append(cc_per_class)
    
    # Example visualization of the first 5 predictions
    if print_stats:
        print("\nValidation Predictions (True Label | Prediction Set):")
        for i in range(5):
            print(f"  {pred_val_labels[i]} | {prediction_sets[i]}")

    res_metrics = (q_hat, coverage, avg_set_size, counts, cc_per_class)

    return prediction_sets, y_calib_pred_relab, y_val_pred_relab, res_metrics

def conformal_prediction_knn_with_synthetic_data(knn_model, calib_data, val_data, alpha=0.1, print_stats=False):
    calib_emb = calib_data[0]
    calib_labels = calib_data[1]
    val_emb = val_data[0]
    val_labels = val_data[1]
    
    len_calib_data = calib_emb.shape[0]
    len_val_data = val_emb.shape[0]


    ############################# Prediction
    y_calib_pred = knn_model.predict(calib_emb)
    #y_calib_pred_relab = np.vectorize(clusters_labels_mapping.get)(y_calib_pred)
    y_val_pred = knn_model.predict(val_emb)
    #y_val_pred_relab = np.vectorize(clusters_labels_mapping.get)(y_val_pred)

    
    ############################### Step 3: Conformal Calibration
    # Compute LOO Non-Conformity Scores (Alpha Values) for Calibration Set
    # This calculation ensures no 0 distance scores are included.
    calib_alphas = calculate_calibration_conformity_scores_loo(calib_emb, y_calib_pred)

    # Calculate Quantile q_hat with Monte Carlo (Bootstrap) Sampling
    CONFIDENCE_LEVEL = 1 - alpha
    n_bootstrap_samples = 500
    q_hat = critical_quantile_mc(calib_alphas, confidence_level=CONFIDENCE_LEVEL, n_bootstrap_samples=n_bootstrap_samples)
    #list_q_hat.append(q_hat)
    
    if print_stats:
        print(f"q_hat values: {q_hat}")

    # Compute Test Set Conformity Scores
    # Distance from test point to the nearest calibration point in each class.
    all_test_scores = calculate_test_conformity_scores(calib_emb, y_calib_pred, val_emb)

    # 5. Determine Prediction Sets
    try:
        prediction_sets, coverage_count, conditional_coverage_total, conditional_coverage_correct = compute_conditional_coverage(q_hat, all_test_scores, y_val_pred)
    except KeyError:
        raise CustomException('Key error in q_hat, continue to next iter to recompute')
        
    # --- 3.4. Evaluation ---
    total_sets = len(prediction_sets)
    coverage = coverage_count / total_sets
    avg_set_size = np.mean([len(s) for s in prediction_sets])
    set_sizes = [len(s) for s in prediction_sets]


    #list_emp_marginal_coverage.append(coverage)
    #list_avg_pred_set_size.append(avg_set_size)
    #list_set_sizes.append(set_sizes)
    
    if print_stats:
        print("\n--- Conformal Prediction Results ---")
        print(f"Target Confidence (1-alpha): {CONFIDENCE_LEVEL:.2f}")
        print(f"Empirical Marginal Coverage: {coverage:.4f}")
        print(f"Average Prediction Set Size: {avg_set_size:.3f}")

    counts = {}
    for item in set_sizes:
        counts[item] = counts.get(item, 0) + 1
    #list_set_sizes_per_class.append(counts)
    if print_stats:
        print(f"Prediction Set Sizes Per Class: {counts}")
        print(f"Proportion of Set Size 1: {counts[1]/sum(counts.values())}")
    
    print("\n--- Empirical Conditional Coverage ---")
    # This shows the actual coverage achieved for each specific class
    
    cc_per_class = {}
    for c in sorted(conditional_coverage_total.keys()):
        total = conditional_coverage_total[c]
        correct = conditional_coverage_correct[c]
        if total > 0:
            cond_cov = correct / total
            cc_per_class[c] = cond_cov
            if print_stats:
                print(f"  Class {c} Coverage (n={total}): {cond_cov:.4f}")
                
        else:
            if print_stats:
                print(f"  Class {c} Coverage (n=0): N/A (No test samples for this class)")
    #list_cc_per_class.append(cc_per_class)
    
    # Example visualization of the first 5 predictions
    if print_stats:
        print("\nValidation Predictions (True Label | Prediction Set):")
        for i in range(5):
            print(f"  {pred_val_labels[i]} | {prediction_sets[i]}")

    res_metrics = (q_hat, coverage, avg_set_size, counts, cc_per_class)

    return prediction_sets, y_calib_pred, y_val_pred, res_metrics