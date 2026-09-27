import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from model.models import EmbeddingsAutoencoderService, EmbeddingsAutoencoder, ContrastiveEmbeddingsTransformer, ContrastiveClusterLearner
from .ethics_contrastive_dataset import setup_ethics_contrastive_training, fetch_subset, extract_from_dataloader
from model.datasets import MilvusDataset
from .synthetic_datasets import generate_synthetic_data, setup_synthetic_dataloaders
from model.loss import LatentPairwiseContrastiveLoss
from model.training import majority_vote_relabeling, majority_vote_relabeling_full

def train_autoencoder_ethics_data(embed_trasformer, milvus_dataset, input_dim=4096, latent_dim=128, 
                            vector_field_name="embedding", device = "cpu", batch_size=32, epochs=10, learning_rate=0.001):
    DEVICE = device
    FEATURE_DIM = input_dim
    LATENT_DIM = latent_dim
    VECTOR_FIELD_NAME = vector_field_name 
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    LEARNING_RATE = learning_rate

    # Dataset
    train_dataset_dataloader = fetch_subset(milvus_dataset, 'train')
    inputs_train, embeddings_train, labels_train = extract_from_dataloader(train_dataset_dataloader)

    train_dataset = TensorDataset(embeddings_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    ## Training function for basic applications such as dimensionality reduction
    model = EmbeddingsAutoencoder(embed_transformer=embed_trasformer, feature_dim=FEATURE_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(EPOCHS):
        for i, batch in enumerate(train_dataloader):
            inputs = batch[0]      # TensorDataset itself returns a tuple or a list of items for each sample
            inputs = inputs.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    return model, train_dataloader

##################################################################

class CustomException(Exception):
    """Custom exception for invalid values."""
    def __init__(self, message="Found invalid value"):
        super().__init__(f"{message}")

def conformal_prediction_with_ethics_data(ccl_model, train_emb, labels_train, calib_emb,
                                        val_emb, device, alpha=0.1, print_stats=False):

    ## Find clusters
    K=2
    clusters_train, centroids_train = ccl_model.cosine_kmeans_fit(train_emb, K)
    clusters_train = clusters_train.detach().cpu()
    # Label matching
    clusters_train, clusters_labels_mapping = majority_vote_relabeling(y_true=labels_train, y_pred=clusters_train)
    #clusters_train, clusters_labels_mapping = majority_vote_relabeling_full(y_true=labels_train, y_pred=clusters_train)
    
    if print_stats:
        print(f"Clusters label mapping: {clusters_labels_mapping}")
    
    # Predict centers
    cluster_labels_valid = ccl_model.predict_clusters(val_emb, centroids_train)
    cluster_labels_valid = cluster_labels_valid.detach().cpu()
    clusters_valid_relabeled = np.vectorize(clusters_labels_mapping.get)(cluster_labels_valid)

    if print_stats:
        print(f"Orig valid clusters: {cluster_labels_valid}")
        print(f"Relabeled valid clusters: {clusters_valid_relabeled}")
    
    cluster_labels_calib = ccl_model.predict_clusters(calib_emb, centroids_train)
    cluster_labels_calib = cluster_labels_calib.detach().cpu()
    clusters_calib_relabeled = np.vectorize(clusters_labels_mapping.get)(cluster_labels_calib)

    if print_stats:
        print(f"Relabled clusters for valid set: {np.unique(clusters_valid_relabeled)}")
        print(f"Relabled clusters for calib set: {np.unique(clusters_calib_relabeled)}")

    ## Predicted labels
    pred_calib_labels = torch.tensor(clusters_calib_relabeled)
    pred_val_labels = torch.tensor(clusters_valid_relabeled)

    if len(pred_calib_labels) < 1:
        raise CustomException('Insufficient predicted labels for calibration, skipping this iteration')
    
    ############################### Step: Conformal Calibration
    # Compute LOO Conformity Scores (Alpha Values) for Calibration Set
    # This calculation ensures no 0 distance scores are included.
    calib_alphas = ccl_model.calculate_calibration_conformity_scores_loo(calib_emb, pred_calib_labels)

    # Calculate Quantile q_hat with Monte Carlo (Bootstrap) Sampling
    CONFIDENCE_LEVEL = 1 - alpha
    n_bootstrap_samples = 500
    q_hat = ccl_model.critical_quantile_mc(calib_alphas, confidence_level=CONFIDENCE_LEVEL, n_bootstrap_samples=n_bootstrap_samples)
    if print_stats:
        print(f"q_hat values: {q_hat}")

    # Compute Test Set Conformity Scores
    # Distance from test point to the nearest calibration point in each class.
    all_test_scores = ccl_model.calculate_test_conformity_scores(calib_emb, pred_calib_labels, val_emb, pred_calib_labels)
    #print(all_test_scores)
    #print(test_labels)

    # 5. Determine Prediction Sets
    try:
        prediction_sets, coverage_count, conditional_coverage_total, conditional_coverage_correct = ccl_model.compute_conditional_coverage(q_hat, all_test_scores, pred_val_labels)
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

    res_metrics = (q_hat, coverage, avg_set_size, counts, cc_per_class)
    return ccl_model, prediction_sets, centroids_train, clusters_labels_mapping, pred_calib_labels, pred_val_labels, res_metrics